# %%
# ONE COLAB CELL: improve EfficientNetB4 from its best RGB-crop checkpoint.
# Upload this file to Colab, paste the whole cell, or run it with %run.

import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------- CONFIGURATION ----------------------------
DATASET_DIR = Path("/content/processed_fundus_dataset/rgb_crop_512_clean_split")
DATASET_ZIP = Path(
    "/content/drive/MyDrive/processed_fundus_dataset/rgb_crop_512.zip"
)
SOURCE_RUN_DIR = Path("/content/drive/MyDrive/dr_runs/b4_rgb_crop")
OUTPUT_DIR = Path("/content/drive/MyDrive/dr_runs/b4_rgb_crop_improved")
RESUME_TRAINING = True
RESUME_CHECKPOINT = None

MODEL_NAME = "efficientnet_b4.ra2_in1k"
IMAGE_SIZE = 384
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
ADDITIONAL_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
HEAD_LR = 2e-5
BACKBONE_LR = 3e-6
MIN_LR = 3e-7
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
ORDINAL_LOSS_WEIGHT = 0.20
NUM_WORKERS = 2
SEED = 42
REQUIRE_GPU = True

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------- COLAB SETUP AND IMPORTS -----------------------
try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    print("Not running in Colab; Google Drive will not be mounted.")

required_packages = ["timm", "scikit-learn", "pandas", "tqdm"]
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", *required_packages]
)

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


# -------------------------- REPRODUCIBILITY ---------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESUME_CHECKPOINT = (
    OUTPUT_DIR / "checkpoint-last.pth"
    if (OUTPUT_DIR / "checkpoint-last.pth").exists()
    else SOURCE_RUN_DIR / "checkpoint-best.pth"
)
print("Resume checkpoint:", RESUME_CHECKPOINT)


# ------------------------ DATASET PREPARATION -------------------------
def has_splits(path):
    if not path.is_dir():
        return False
    names = {item.name.lower() for item in path.iterdir() if item.is_dir()}
    return "train" in names and "test" in names and bool(
        {"val", "valid", "validation"} & names
    )


def find_dataset_root(path):
    if has_splits(path):
        return path
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_dir() and has_splits(candidate):
                return candidate
    raise FileNotFoundError(
        f"Cannot find train/validation/test folders below: {path}"
    )


if not DATASET_DIR.exists():
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR}\n"
            f"Zip not found at {DATASET_ZIP}\n"
            "Upload rgb_crop_512.zip to Drive or change the paths above."
        )
    extract_dir = DATASET_DIR.parent
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {DATASET_ZIP} -> {extract_dir}")
    with zipfile.ZipFile(DATASET_ZIP, "r") as archive:
        archive.extractall(extract_dir)

DATASET_ROOT = find_dataset_root(DATASET_DIR.parent if not has_splits(DATASET_DIR) else DATASET_DIR)
split_dirs = {item.name.lower(): item for item in DATASET_ROOT.iterdir() if item.is_dir()}
TRAIN_DIR = split_dirs["train"]
VAL_DIR = next(split_dirs[name] for name in ("validation", "valid", "val") if name in split_dirs)
TEST_DIR = split_dirs["test"]

print("Dataset root:", DATASET_ROOT)
print("Output:", OUTPUT_DIR)


# ------------------------- TRANSFORMATIONS ----------------------------
# RGB crop was already applied offline. Do not apply CLAHE or Ben Graham here.
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=10,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

expected_classes = [str(i) for i in range(NUM_CLASSES)]
if train_dataset.classes != expected_classes:
    raise ValueError(
        f"Expected class folders {expected_classes}, found {train_dataset.classes}"
    )
if val_dataset.class_to_idx != train_dataset.class_to_idx:
    raise ValueError("Validation class folders do not match training class folders.")
if test_dataset.class_to_idx != train_dataset.class_to_idx:
    raise ValueError("Test class folders do not match training class folders.")

train_targets = np.asarray(train_dataset.targets)
class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if REQUIRE_GPU and device.type != "cuda":
    raise RuntimeError(
        "GPU is not enabled. In Colab select Runtime > Change runtime type > "
        "T4 GPU, then run this cell again."
    )
pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    drop_last=True,
    persistent_workers=NUM_WORKERS > 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=NUM_WORKERS > 0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=NUM_WORKERS > 0,
)

print("Device:", device)
print("Classes:", train_dataset.class_to_idx)
print("Train/val/test:", len(train_dataset), len(val_dataset), len(test_dataset))
print("Train class counts:", class_counts.tolist())


# ---------------------------- MODEL SETUP -----------------------------
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=0.25,
    drop_path_rate=0.15,
)
model = model.to(device)


def is_classifier_parameter(name):
    return name.startswith(("classifier", "fc", "head"))


def create_optimizer():
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_classifier_parameter(name):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    groups = [
        {"params": backbone_parameters, "lr": BACKBONE_LR},
        {"params": head_parameters, "lr": HEAD_LR},
    ]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)


def combined_loss(logits, targets):
    classification_loss = criterion(logits.float(), targets)
    probabilities = logits.float().softmax(dim=1)
    predicted_cdf = probabilities.cumsum(dim=1)[:, :-1]
    true_distribution = torch.nn.functional.one_hot(
        targets, num_classes=NUM_CLASSES
    ).float()
    true_cdf = true_distribution.cumsum(dim=1)[:, :-1]
    ordinal_loss = torch.mean((predicted_cdf - true_cdf) ** 2)
    return (
        classification_loss + ORDINAL_LOSS_WEIGHT * ordinal_loss,
        classification_loss.detach(),
        ordinal_loss.detach(),
    )


# ----------------------- TRAINING AND METRICS -------------------------
def calculate_metrics(targets, predictions):
    report = classification_report(
        targets,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "qwk": float(
            cohen_kappa_score(targets, predictions, weights="quadratic")
        ),
        "per_class_recall": {
            name: float(report[name]["recall"]) for name in CLASS_NAMES
        },
    }


def train_one_epoch(loader, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_ce_loss = 0.0
    running_ordinal_loss = 0.0
    seen = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for step, (images, targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss, ce_loss, ordinal_loss = combined_loss(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {step}: {loss.item()}"
            )

        scaler.scale(loss / ACCUMULATION_STEPS).backward()
        should_step = (
            (step + 1) % ACCUMULATION_STEPS == 0
            or step + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * targets.size(0)
        running_ce_loss += ce_loss.item() * targets.size(0)
        running_ordinal_loss += ordinal_loss.item() * targets.size(0)
        seen += targets.size(0)
        progress.set_postfix(loss=f"{running_loss / seen:.4f}")

    return {
        "loss": running_loss / max(seen, 1),
        "ce_loss": running_ce_loss / max(seen, 1),
        "ordinal_loss": running_ordinal_loss / max(seen, 1),
    }


@torch.inference_mode()
def evaluate(loader):
    model.eval()
    running_loss = 0.0
    seen = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, targets in tqdm(loader, desc="Evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
        loss, _, _ = combined_loss(logits.float(), targets)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss detected.")
        probabilities = logits.float().softmax(dim=1)
        running_loss += loss.item() * targets.size(0)
        seen += targets.size(0)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
        all_probabilities.extend(probabilities.float().cpu().tolist())

    metrics = calculate_metrics(all_targets, all_predictions)
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics, all_targets, all_predictions, all_probabilities


def save_checkpoint(
    path,
    epoch,
    best_qwk,
    fine_tune_start_epoch,
    fine_tune_target_epoch,
    optimizer=None,
    scheduler=None,
):
    payload = {
        "model": model.state_dict(),
        "architecture": MODEL_NAME,
        "epoch": epoch,
        "best_qwk": best_qwk,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_to_idx": train_dataset.class_to_idx,
        "preprocessing": "offline_rgb_crop_clean_split",
        "fine_tune_start_epoch": fine_tune_start_epoch,
        "fine_tune_target_epoch": fine_tune_target_epoch,
        "ordinal_loss_weight": ORDINAL_LOSS_WEIGHT,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "normalization_mean": (0.485, 0.456, 0.406),
        "normalization_std": (0.229, 0.224, 0.225),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if amp_enabled:
        payload["scaler"] = scaler.state_dict()
    torch.save(
        payload,
        path,
    )


history_path = OUTPUT_DIR / "history.jsonl"
best_qwk = -1.0
epochs_without_improvement = 0
start_epoch = 0
fine_tune_start_epoch = 0
fine_tune_target_epoch = 0

resume_checkpoint = None
resuming_improved_run = (
    RESUME_TRAINING
    and RESUME_CHECKPOINT.exists()
    and RESUME_CHECKPOINT.parent == OUTPUT_DIR
)
if RESUME_TRAINING and RESUME_CHECKPOINT.exists():
    resume_checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    if resume_checkpoint.get("architecture") != MODEL_NAME:
        raise ValueError(
            f"Checkpoint architecture {resume_checkpoint.get('architecture')} "
            f"does not match MODEL_NAME={MODEL_NAME}"
        )
    if resume_checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError(
            f"Checkpoint num_classes {resume_checkpoint.get('num_classes')} "
            f"does not match NUM_CLASSES={NUM_CLASSES}"
        )
    model.load_state_dict(resume_checkpoint["model"])
    start_epoch = int(resume_checkpoint.get("epoch", 0))
    best_qwk = float(resume_checkpoint.get("best_qwk", -1.0))
    if resuming_improved_run:
        fine_tune_start_epoch = int(
            resume_checkpoint.get("fine_tune_start_epoch", start_epoch)
        )
        fine_tune_target_epoch = int(
            resume_checkpoint.get(
                "fine_tune_target_epoch",
                fine_tune_start_epoch + ADDITIONAL_EPOCHS,
            )
        )
    else:
        fine_tune_start_epoch = start_epoch
        fine_tune_target_epoch = start_epoch + ADDITIONAL_EPOCHS
    print(
        f"Resumed from {RESUME_CHECKPOINT}: "
        f"epoch={start_epoch}, best_qwk={best_qwk:.4f}"
    )
else:
    raise FileNotFoundError(
        f"Cannot continue B4 because checkpoint was not found: "
        f"{RESUME_CHECKPOINT}"
    )

trainable = sum(parameter.numel() for parameter in model.parameters())
print(f"Backbone frozen=False; trainable parameters={trainable:,}")
print(
    f"Fine-tuning epochs {start_epoch + 1}..{fine_tune_target_epoch}; "
    f"baseline QWK={best_qwk:.4f}"
)

optimizer = create_optimizer()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(fine_tune_target_epoch - fine_tune_start_epoch, 1),
    eta_min=MIN_LR,
    last_epoch=-1,
)

if resuming_improved_run and "optimizer" in resume_checkpoint:
    try:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        print("Resumed optimizer state.")
    except ValueError as error:
        print(f"Could not resume optimizer state; using a new optimizer. {error}")

if resuming_improved_run and "scheduler" in resume_checkpoint:
    scheduler.load_state_dict(resume_checkpoint["scheduler"])
    print("Resumed scheduler state.")

if (
    resuming_improved_run
    and "scaler" in resume_checkpoint
    and amp_enabled
):
    scaler.load_state_dict(resume_checkpoint["scaler"])
    print("Resumed AMP scaler state.")

if not (OUTPUT_DIR / "checkpoint-best.pth").exists():
    save_checkpoint(
        OUTPUT_DIR / "checkpoint-best.pth",
        start_epoch,
        best_qwk,
        fine_tune_start_epoch,
        fine_tune_target_epoch,
    )
    print("Copied baseline weights into the improved run directory.")

for epoch in range(start_epoch, fine_tune_target_epoch):
    train_metrics = train_one_epoch(train_loader, optimizer)
    val_metrics, _, _, _ = evaluate(val_loader)
    record = {
        "epoch": epoch + 1,
        "backbone_lr": optimizer.param_groups[0]["lr"],
        "head_lr": optimizer.param_groups[1]["lr"],
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    print(json.dumps(record, ensure_ascii=False))
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    if val_metrics["qwk"] > best_qwk + 1e-4:
        best_qwk = val_metrics["qwk"]
        epochs_without_improvement = 0
        save_checkpoint(
            OUTPUT_DIR / "checkpoint-best.pth",
            epoch + 1,
            best_qwk,
            fine_tune_start_epoch,
            fine_tune_target_epoch,
            optimizer,
            scheduler,
        )
        print(f"Saved best checkpoint: QWK={best_qwk:.4f}")
    else:
        epochs_without_improvement += 1

    scheduler.step()
    save_checkpoint(
        OUTPUT_DIR / "checkpoint-last.pth",
        epoch + 1,
        best_qwk,
        fine_tune_start_epoch,
        fine_tune_target_epoch,
        optimizer,
        scheduler,
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("Early stopping.")
        break


# ------------------------- FINAL TEST REPORT --------------------------
checkpoint = torch.load(
    OUTPUT_DIR / "checkpoint-best.pth",
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model"])
test_metrics, targets, predictions, probabilities = evaluate(test_loader)

(OUTPUT_DIR / "test_metrics.json").write_text(
    json.dumps(test_metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

prediction_rows = []
for index, (path, _) in enumerate(test_dataset.samples):
    row = {
        "image_path": os.fspath(Path(path).relative_to(DATASET_ROOT)),
        "true_grade": targets[index],
        "predicted_grade": predictions[index],
    }
    for grade in range(NUM_CLASSES):
        row[f"probability_{grade}"] = probabilities[index][grade]
    prediction_rows.append(row)
pd.DataFrame(prediction_rows).to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)

matrix = confusion_matrix(targets, predictions, labels=list(range(NUM_CLASSES)))
pd.DataFrame(
    matrix,
    index=[f"true_{name}" for name in CLASS_NAMES],
    columns=[f"pred_{name}" for name in CLASS_NAMES],
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

print("\nFinal test metrics:")
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
print("Best checkpoint:", OUTPUT_DIR / "checkpoint-best.pth")
print("All outputs:", OUTPUT_DIR)


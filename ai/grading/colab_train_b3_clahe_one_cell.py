# %%
# ONE COLAB CELL: EfficientNetB3 on the clean CLAHE split.
# Stage 1 trains at 384px; stage 2 fine-tunes the best model at 512px.

import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------- CONFIGURATION ----------------------------
DATASET_DIR = Path(
    "/content/processed_fundus_dataset/clahe_lab_512_clean_split"
)
DATASET_ZIP = Path(
    "/content/drive/MyDrive/processed_fundus_dataset/"
    "clahe_lab_512_split.zip"
)
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/dr_runs/b3_clahe_improved"
)

MODEL_NAME = "efficientnet_b3.ra2_in1k"
BATCH_SIZE_384 = 8
BATCH_SIZE_512 = 4
ACCUMULATION_STEPS = 2
NUM_WORKERS = 2
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
ORDINAL_LOSS_WEIGHT = 0.20
CLASS_WEIGHTS = [1.0, 1.15, 1.25, 1.0, 1.0]
MIN_LR = 3e-7
SEED = 42
REQUIRE_GPU = True
RESUME_TRAINING = True

STAGES = [
    {
        "name": "train_384",
        "image_size": 384,
        "batch_size": BATCH_SIZE_384,
        "epochs": 20,
        "freeze_epochs": 2,
        "backbone_lr": 2e-5,
        "head_lr": 1e-4,
        "patience": 5,
    },
    {
        "name": "finetune_512",
        "image_size": 512,
        "batch_size": BATCH_SIZE_512,
        "epochs": 5,
        "freeze_epochs": 0,
        "backbone_lr": 2e-6,
        "head_lr": 1e-5,
        "patience": 2,
    },
]

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------- COLAB SETUP AND IMPORTS -----------------------
try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    print("Not running in Colab; Google Drive will not be mounted.")

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "timm",
        "scikit-learn",
        "pandas",
        "tqdm",
    ]
)

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if REQUIRE_GPU and device.type != "cuda":
    raise RuntimeError(
        "GPU is not enabled. In Colab select Runtime > Change runtime type > "
        "T4 GPU, then run this cell again."
    )
amp_enabled = device.type == "cuda"
pin_memory = amp_enabled
print("Device:", device)
if amp_enabled:
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------ DATASET PREPARATION -------------------------
def has_splits(path):
    if not path.is_dir():
        return False
    names = {item.name.lower() for item in path.iterdir() if item.is_dir()}
    return "train" in names and "test" in names and bool(
        {"validation", "valid", "val"} & names
    )


def find_dataset_root(path):
    if has_splits(path):
        return path
    candidates = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_dir() and has_splits(item)
        ),
        key=lambda item: len(item.parts),
    )
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find train/validation/test folders below {path}"
        )
    expected = [
        item for item in candidates if "clahe_lab_512_split" in item.name
    ]
    return expected[0] if expected else candidates[0]


if not has_splits(DATASET_DIR):
    if not DATASET_ZIP.is_file():
        raise FileNotFoundError(
            f"Dataset zip was not found: {DATASET_ZIP}\n"
            "Upload clahe_lab_512_split.zip to that Drive path."
        )
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {DATASET_ZIP} -> {DATASET_DIR.parent}")
    with zipfile.ZipFile(DATASET_ZIP, "r") as archive:
        archive.extractall(DATASET_DIR.parent)

DATASET_ROOT = find_dataset_root(DATASET_DIR.parent)
split_dirs = {
    item.name.lower(): item
    for item in DATASET_ROOT.iterdir()
    if item.is_dir()
}
TRAIN_DIR = split_dirs["train"]
VAL_DIR = next(
    split_dirs[name]
    for name in ("validation", "valid", "val")
    if name in split_dirs
)
TEST_DIR = split_dirs["test"]
print("Dataset root:", DATASET_ROOT)
print("Output:", OUTPUT_DIR)


def build_transforms(image_size, training):
    operations = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
        )
    ]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=8 if image_size >= 512 else 10,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.02, 0.02),
                    scale=(0.96, 1.04),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(
                    brightness=0.04,
                    contrast=0.04,
                    saturation=0.03,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


def create_loaders(stage):
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=build_transforms(stage["image_size"], training=True),
    )
    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=build_transforms(stage["image_size"], training=False),
    )
    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=build_transforms(stage["image_size"], training=False),
    )
    expected_classes = [str(index) for index in range(NUM_CLASSES)]
    if train_dataset.classes != expected_classes:
        raise ValueError(
            f"Expected class folders {expected_classes}, "
            f"found {train_dataset.classes}"
        )
    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError("Validation classes do not match training classes.")
    if test_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError("Test classes do not match training classes.")

    common = {
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory,
        "persistent_workers": NUM_WORKERS > 0,
    }
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        DataLoader(
            train_dataset,
            batch_size=stage["batch_size"],
            shuffle=True,
            drop_last=True,
            **common,
        ),
        DataLoader(
            val_dataset,
            batch_size=stage["batch_size"],
            shuffle=False,
            **common,
        ),
        DataLoader(
            test_dataset,
            batch_size=stage["batch_size"],
            shuffle=False,
            **common,
        ),
    )


# ---------------------------- MODEL SETUP -----------------------------
resume_path = OUTPUT_DIR / "checkpoint-last.pth"
resume_checkpoint = None
if RESUME_TRAINING and resume_path.is_file():
    resume_checkpoint = torch.load(
        resume_path, map_location="cpu", weights_only=False
    )

model = timm.create_model(
    MODEL_NAME,
    pretrained=resume_checkpoint is None,
    num_classes=NUM_CLASSES,
    drop_rate=0.25,
    drop_path_rate=0.15,
).to(device)

if resume_checkpoint is not None:
    if resume_checkpoint.get("architecture") != MODEL_NAME:
        raise ValueError("Resume checkpoint architecture does not match B3.")
    model.load_state_dict(resume_checkpoint["model"], strict=True)
    print(
        f"Resumed {resume_path}: stage={resume_checkpoint['stage_name']}, "
        f"stage_epoch={resume_checkpoint['stage_epoch']}, "
        f"best_qwk={resume_checkpoint['best_qwk']:.4f}"
    )
    next_stage_epoch = int(resume_checkpoint["stage_epoch"]) + 1
    if int(resume_checkpoint["stage_index"]) == 0:
        print(
            f"Resume plan: continue train_384 at epoch "
            f"{next_stage_epoch}/{STAGES[0]['epochs']}, then load the global "
            "best checkpoint and fine-tune at 512px."
        )
    else:
        print(
            f"Resume plan: continue finetune_512 at epoch "
            f"{next_stage_epoch}/{STAGES[1]['epochs']}."
        )
else:
    print("Starting a clean run from ImageNet-pretrained EfficientNetB3.")


def is_classifier_parameter(name):
    return name.startswith(("classifier", "fc", "head"))


def set_backbone_frozen(frozen):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            is_classifier_parameter(name) or not frozen
        )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Backbone frozen={frozen}; trainable={trainable:,}")


def create_optimizer(stage, backbone_frozen):
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_classifier_parameter(name):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    groups = [
        {
            "params": head_parameters,
            "lr": stage["head_lr"],
            "group_name": "head",
        }
    ]
    if not backbone_frozen:
        groups.insert(
            0,
            {
                "params": backbone_parameters,
                "lr": stage["backbone_lr"],
                "group_name": "backbone",
            },
        )
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def apply_cosine_lr(optimizer, stage, epoch_index):
    trainable_epochs = max(
        stage["epochs"] - stage["freeze_epochs"], 1
    )
    progress = max(
        epoch_index - stage["freeze_epochs"], 0
    ) / max(trainable_epochs - 1, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    for group in optimizer.param_groups:
        base_lr = (
            stage["backbone_lr"]
            if group["group_name"] == "backbone"
            else stage["head_lr"]
        )
        group["lr"] = MIN_LR + (base_lr - MIN_LR) * cosine


class_weight_tensor = torch.tensor(
    CLASS_WEIGHTS, dtype=torch.float32, device=device
)
criterion = nn.CrossEntropyLoss(
    weight=class_weight_tensor,
    label_smoothing=LABEL_SMOOTHING,
)
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)


def combined_loss(logits, targets):
    logits = logits.float()
    ce_loss = criterion(logits, targets)
    probabilities = logits.softmax(dim=1)
    predicted_cdf = probabilities.cumsum(dim=1)[:, :-1]
    true_distribution = F.one_hot(
        targets, num_classes=NUM_CLASSES
    ).float()
    true_cdf = true_distribution.cumsum(dim=1)[:, :-1]
    ordinal_loss = torch.mean((predicted_cdf - true_cdf) ** 2)
    total_loss = ce_loss + ORDINAL_LOSS_WEIGHT * ordinal_loss
    return total_loss, ce_loss.detach(), ordinal_loss.detach()


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
            f1_score(
                targets, predictions, average="macro", zero_division=0
            )
        ),
        "qwk": float(
            cohen_kappa_score(targets, predictions, weights="quadratic")
        ),
        "per_class_recall": {
            name: float(report[name]["recall"])
            for name in CLASS_NAMES
        },
    }


def train_one_epoch(loader, optimizer, backbone_frozen):
    model.train()
    # Preserve pretrained BatchNorm statistics. Updating them while the
    # backbone is frozen can make the first validation pass non-finite.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "ce_loss": 0.0, "ordinal_loss": 0.0}
    seen = 0
    use_amp = amp_enabled and not backbone_frozen
    progress = tqdm(loader, desc="Train", leave=False)

    for step, (images, targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss, ce_loss, ordinal_loss = combined_loss(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite train loss at step {step}")

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

        batch_size = targets.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["ce_loss"] += ce_loss.item() * batch_size
        totals["ordinal_loss"] += ordinal_loss.item() * batch_size
        seen += batch_size
        progress.set_postfix(loss=f"{totals['loss'] / seen:.4f}")

    return {key: value / max(seen, 1) for key, value in totals.items()}


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
        # Validation is intentionally FP32. Some T4/FP16 runs produce finite
        # train loss but overflow logits on the first evaluation batch.
        with torch.amp.autocast("cuda", enabled=False):
            logits = model(images.float())
        if not torch.isfinite(logits).all():
            finite_ratio = torch.isfinite(logits).float().mean().item()
            raise RuntimeError(
                "Model produced non-finite FP32 validation logits "
                f"(finite ratio={finite_ratio:.4f})."
            )
        loss, _, _ = combined_loss(logits.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss detected.")

        probabilities = logits.float().softmax(dim=1)
        running_loss += loss.item() * targets.size(0)
        seen += targets.size(0)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    metrics = calculate_metrics(all_targets, all_predictions)
    metrics["loss"] = running_loss / max(seen, 1)
    return (
        metrics,
        np.asarray(all_targets, dtype=np.int64),
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
    )


def save_checkpoint(
    path,
    stage_index,
    stage_epoch,
    global_epoch,
    best_qwk,
    image_size,
    no_improvement,
    optimizer=None,
):
    payload = {
        "model": model.state_dict(),
        "architecture": MODEL_NAME,
        "stage_index": stage_index,
        "stage_name": STAGES[stage_index]["name"],
        "stage_epoch": stage_epoch,
        "global_epoch": global_epoch,
        "epoch": global_epoch,
        "best_qwk": best_qwk,
        "no_improvement": no_improvement,
        "image_size": image_size,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_to_idx": {str(index): index for index in range(NUM_CLASSES)},
        "preprocessing": "offline_clahe_lab_clean_split",
        "class_weights": CLASS_WEIGHTS,
        "ordinal_loss_weight": ORDINAL_LOSS_WEIGHT,
        "normalization_mean": (0.485, 0.456, 0.406),
        "normalization_std": (0.229, 0.224, 0.225),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if amp_enabled:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


history_path = OUTPUT_DIR / "history.jsonl"
best_qwk = (
    float(resume_checkpoint.get("best_qwk", -1.0))
    if resume_checkpoint is not None
    else -1.0
)
global_epoch = (
    int(resume_checkpoint.get("global_epoch", 0))
    if resume_checkpoint is not None
    else 0
)
resume_stage_index = (
    int(resume_checkpoint.get("stage_index", 0))
    if resume_checkpoint is not None
    else 0
)
resume_stage_epoch = (
    int(resume_checkpoint.get("stage_epoch", 0))
    if resume_checkpoint is not None
    else 0
)

for stage_index, stage in enumerate(STAGES):
    if stage_index < resume_stage_index:
        continue

    start_stage_epoch = (
        resume_stage_epoch if stage_index == resume_stage_index else 0
    )
    if start_stage_epoch == 0 and stage_index > 0:
        best_checkpoint = torch.load(
            OUTPUT_DIR / "checkpoint-best.pth",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model"], strict=True)
        print(
            f"Loaded global best QWK={best_checkpoint['best_qwk']:.4f} "
            f"before {stage['name']}."
        )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_loaders(stage)
    class_counts = np.bincount(
        np.asarray(train_dataset.targets), minlength=NUM_CLASSES
    )
    print(
        f"\nStage {stage_index + 1}/{len(STAGES)}: {stage['name']} | "
        f"size={stage['image_size']} | batch={stage['batch_size']}"
    )
    print("Train/val/test:", len(train_dataset), len(val_dataset), len(test_dataset))
    print("Train class counts:", class_counts.tolist())

    no_improvement = (
        int(resume_checkpoint.get("no_improvement", 0))
        if resume_checkpoint is not None
        and stage_index == resume_stage_index
        else 0
    )
    current_frozen = None
    optimizer = None

    for stage_epoch in range(start_stage_epoch, stage["epochs"]):
        frozen = stage_epoch < stage["freeze_epochs"]
        if optimizer is None or frozen != current_frozen:
            set_backbone_frozen(frozen)
            optimizer = create_optimizer(stage, frozen)
            current_frozen = frozen
            if (
                resume_checkpoint is not None
                and stage_index == resume_stage_index
                and stage_epoch == start_stage_epoch
                and "optimizer" in resume_checkpoint
            ):
                try:
                    optimizer.load_state_dict(
                        resume_checkpoint["optimizer"]
                    )
                    print("Resumed optimizer state.")
                except ValueError:
                    print("Optimizer shape changed; using a fresh optimizer.")
                if "scaler" in resume_checkpoint and amp_enabled:
                    scaler.load_state_dict(resume_checkpoint["scaler"])

        apply_cosine_lr(optimizer, stage, stage_epoch)
        train_metrics = train_one_epoch(
            train_loader,
            optimizer,
            backbone_frozen=frozen,
        )
        val_metrics, _, _, _ = evaluate(val_loader)
        global_epoch += 1
        record = {
            "global_epoch": global_epoch,
            "stage": stage["name"],
            "stage_epoch": stage_epoch + 1,
            "image_size": stage["image_size"],
            "learning_rates": {
                group["group_name"]: group["lr"]
                for group in optimizer.param_groups
            },
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
        }
        print(json.dumps(record, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if val_metrics["qwk"] > best_qwk + 1e-4:
            best_qwk = val_metrics["qwk"]
            no_improvement = 0
            save_checkpoint(
                OUTPUT_DIR / "checkpoint-best.pth",
                stage_index,
                stage_epoch + 1,
                global_epoch,
                best_qwk,
                stage["image_size"],
                no_improvement,
                optimizer,
            )
            print(f"Saved best checkpoint: QWK={best_qwk:.4f}")
        else:
            no_improvement += 1

        save_checkpoint(
            OUTPUT_DIR / "checkpoint-last.pth",
            stage_index,
            stage_epoch + 1,
            global_epoch,
            best_qwk,
            stage["image_size"],
            no_improvement,
            optimizer,
        )
        if no_improvement >= stage["patience"]:
            print(f"Early stopping {stage['name']}.")
            break

    resume_checkpoint = None
    resume_stage_epoch = 0


# ---------------- THRESHOLD CALIBRATION AND FINAL TEST ----------------
best_checkpoint = torch.load(
    OUTPUT_DIR / "checkpoint-best.pth",
    map_location=device,
    weights_only=False,
)
model.load_state_dict(best_checkpoint["model"], strict=True)
final_stage = STAGES[-1]
(
    _,
    val_dataset,
    test_dataset,
    _,
    val_loader,
    test_loader,
) = create_loaders(final_stage)

val_metrics_raw, val_targets, val_predictions_raw, val_probabilities = evaluate(
    val_loader
)


def expected_grades(probabilities):
    grades = np.arange(NUM_CLASSES, dtype=np.float32)
    return probabilities @ grades


def threshold_predictions(scores, thresholds):
    return np.digitize(scores, thresholds).astype(np.int64)


def calibration_score(targets, predictions):
    qwk = cohen_kappa_score(targets, predictions, weights="quadratic")
    macro_f1 = f1_score(
        targets, predictions, average="macro", zero_division=0
    )
    return float(qwk + 0.10 * macro_f1)


def optimize_thresholds(targets, scores):
    thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    candidate_values = np.arange(0.20, 3.81, 0.02)
    for _ in range(6):
        changed = False
        for index in range(4):
            lower = thresholds[index - 1] + 0.02 if index > 0 else 0.05
            upper = (
                thresholds[index + 1] - 0.02
                if index < 3
                else 3.95
            )
            candidates = candidate_values[
                (candidate_values >= lower)
                & (candidate_values <= upper)
            ]
            best_value = thresholds[index]
            best_score = calibration_score(
                targets, threshold_predictions(scores, thresholds)
            )
            for candidate in candidates:
                trial = thresholds.copy()
                trial[index] = candidate
                score = calibration_score(
                    targets, threshold_predictions(scores, trial)
                )
                if score > best_score + 1e-9:
                    best_score = score
                    best_value = candidate
            if abs(best_value - thresholds[index]) > 1e-9:
                thresholds[index] = best_value
                changed = True
        if not changed:
            break
    return thresholds


val_scores = expected_grades(val_probabilities)
thresholds = optimize_thresholds(val_targets, val_scores)
val_predictions_calibrated = threshold_predictions(val_scores, thresholds)
val_metrics_calibrated = calculate_metrics(
    val_targets, val_predictions_calibrated
)

test_metrics_raw, test_targets, test_predictions_raw, test_probabilities = evaluate(
    test_loader
)
test_scores = expected_grades(test_probabilities)
test_predictions_calibrated = threshold_predictions(
    test_scores, thresholds
)
test_metrics_calibrated = calculate_metrics(
    test_targets, test_predictions_calibrated
)

threshold_payload = {
    "thresholds": thresholds.tolist(),
    "optimized_on": "validation",
    "objective": "qwk + 0.10 * macro_f1",
    "validation_raw": val_metrics_raw,
    "validation_calibrated": val_metrics_calibrated,
}
(OUTPUT_DIR / "calibrated_thresholds.json").write_text(
    json.dumps(threshold_payload, indent=2),
    encoding="utf-8",
)

final_metrics = {
    "best_checkpoint_qwk": float(best_checkpoint["best_qwk"]),
    "best_checkpoint_image_size": int(best_checkpoint["image_size"]),
    "thresholds": thresholds.tolist(),
    "test_raw_argmax": test_metrics_raw,
    "test_calibrated": test_metrics_calibrated,
}
(OUTPUT_DIR / "test_metrics.json").write_text(
    json.dumps(final_metrics, indent=2),
    encoding="utf-8",
)

prediction_rows = []
for index, (path, _) in enumerate(test_dataset.samples):
    row = {
        "image_path": os.fspath(Path(path).relative_to(DATASET_ROOT)),
        "true_grade": int(test_targets[index]),
        "raw_prediction": int(test_predictions_raw[index]),
        "expected_grade": float(test_scores[index]),
        "calibrated_prediction": int(
            test_predictions_calibrated[index]
        ),
    }
    for grade in range(NUM_CLASSES):
        row[f"probability_{grade}"] = float(
            test_probabilities[index, grade]
        )
    prediction_rows.append(row)
pd.DataFrame(prediction_rows).to_csv(
    OUTPUT_DIR / "test_predictions.csv", index=False
)

for name, predictions in (
    ("raw", test_predictions_raw),
    ("calibrated", test_predictions_calibrated),
):
    matrix = confusion_matrix(
        test_targets, predictions, labels=list(range(NUM_CLASSES))
    )
    pd.DataFrame(
        matrix,
        index=[f"true_{class_name}" for class_name in CLASS_NAMES],
        columns=[f"pred_{class_name}" for class_name in CLASS_NAMES],
    ).to_csv(OUTPUT_DIR / f"confusion_matrix_{name}.csv")

per_grade_rows = []
for grade, class_name in enumerate(CLASS_NAMES):
    mask = test_targets == grade
    total = int(mask.sum())
    raw_correct = int((test_predictions_raw[mask] == grade).sum())
    calibrated_correct = int(
        (test_predictions_calibrated[mask] == grade).sum()
    )
    per_grade_rows.append(
        {
            "grade": grade,
            "class_name": class_name,
            "total_images": total,
            "raw_correct": raw_correct,
            "raw_recall_percent": round(
                100.0 * raw_correct / total if total else 0.0, 2
            ),
            "calibrated_correct": calibrated_correct,
            "calibrated_recall_percent": round(
                100.0 * calibrated_correct / total if total else 0.0,
                2,
            ),
        }
    )
per_grade_table = pd.DataFrame(per_grade_rows)
per_grade_table.to_csv(
    OUTPUT_DIR / "test_per_grade.csv", index=False
)

print("\nCalibrated thresholds:", thresholds.tolist())
print("\nRaw test metrics:")
print(json.dumps(test_metrics_raw, indent=2))
print("\nCalibrated test metrics:")
print(json.dumps(test_metrics_calibrated, indent=2))
print("\nTest recall by grade:")
print(per_grade_table.to_string(index=False))
print("\nBest checkpoint:", OUTPUT_DIR / "checkpoint-best.pth")
print("All outputs:", OUTPUT_DIR)

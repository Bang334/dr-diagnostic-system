"""Train U-Net-family models for IDRiD MA/HE/EX lesion segmentation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ai.segmentation.data import (
    IDRiDLesionDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LESION_NAMES,
    create_idrid_splits,
)
from ai.segmentation.metrics import SegmentationMeter, calibrate_dice_thresholds


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--architecture",
        choices=("unet", "unetplusplus", "manet"),
        default="unet",
    )
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument(
        "--encoder-weights",
        default="imagenet",
        help="SMP encoder weights such as imagenet; use 'none' for random init",
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--focal-weight", type=float, default=0.4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "image_size",
        "epochs",
        "patience",
        "batch_size",
        "accum_steps",
        "num_workers",
    ):
        if getattr(args, name) < 1 and name != "num_workers":
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.freeze_epochs < 0:
        raise ValueError("num_workers and freeze_epochs cannot be negative")
    if args.preview_count < 0:
        raise ValueError("preview_count cannot be negative")
    if args.freeze_epochs >= args.epochs:
        raise ValueError("freeze_epochs must be smaller than epochs")
    if not 0.0 < args.focal_alpha < 1.0:
        raise ValueError("focal_alpha must be between 0 and 1")
    if args.dice_weight < 0 or args.focal_weight < 0:
        raise ValueError("Loss weights cannot be negative")
    if args.dice_weight + args.focal_weight <= 0:
        raise ValueError("At least one loss weight must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new run name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_model(args: argparse.Namespace) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "Install ai/segmentation/requirements.txt before training"
        ) from exc
    constructors = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "manet": smp.MAnet,
    }
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    return constructors[args.architecture](
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=len(LESION_NAMES),
        activation=None,
    )


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        *,
        alpha: float,
        gamma: float,
        dice_weight: float,
        focal_weight: float,
    ) -> None:
        super().__init__()
        total = dice_weight + focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight / total
        self.focal_weight = focal_weight / total

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probabilities = torch.sigmoid(logits)
        reduce_dims = (0, 2, 3)
        intersection = (probabilities * targets).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def create_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def _autocast(enabled: bool) -> Any:
    return torch.amp.autocast("cuda", enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    accum_steps: int,
    amp_enabled: bool,
    encoder_frozen: bool,
) -> float:
    model.train()
    if encoder_frozen:
        model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    for step, (images, masks, _) in enumerate(tqdm(loader, desc="lesion-train", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with _autocast(amp_enabled):
            loss = criterion(model(images), masks)
            scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    thresholds: Sequence[float],
    amp_enabled: bool,
    collect: bool,
    description: str,
) -> Tuple[float, Dict[str, Any], List[torch.Tensor], List[torch.Tensor]]:
    model.eval()
    meter = SegmentationMeter(thresholds)
    losses: List[float] = []
    probability_batches: List[torch.Tensor] = []
    target_batches: List[torch.Tensor] = []
    for images, masks, _ in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with _autocast(amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
        probabilities = torch.sigmoid(logits).float().cpu()
        targets = masks.float().cpu()
        meter.update(probabilities, targets)
        if collect:
            probability_batches.append(probabilities)
            target_batches.append(targets)
        losses.append(float(loss))
    return sum(losses) / len(losses), meter.compute(), probability_batches, target_batches


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    *,
    epoch: int,
    best_val_dice: float,
    thresholds: Sequence[float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "args": {
            key: os.fspath(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "epoch": epoch,
        "best_val_macro_dice": float(best_val_dice),
        "lesion_names": list(LESION_NAMES),
        "thresholds": [float(value) for value in thresholds],
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


@torch.no_grad()
def save_previews(
    model: nn.Module,
    dataset: IDRiDLesionDataset,
    output_dir: Path,
    device: torch.device,
    *,
    thresholds: Sequence[float],
    count: int,
    amp_enabled: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(((255, 40, 40), (220, 40, 220), (255, 220, 30)), dtype=np.float32)
    mean = IMAGENET_MEAN.reshape(1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 3)
    model.eval()
    for index in range(min(count, len(dataset))):
        image, target, image_id = dataset[index]
        with _autocast(amp_enabled):
            logits = model(image.unsqueeze(0).to(device))
        probability = torch.sigmoid(logits)[0].float().cpu().numpy()
        prediction = probability >= np.asarray(thresholds)[:, None, None]
        image_rgb = image.numpy().transpose(1, 2, 0)
        image_rgb = np.clip((image_rgb * std + mean) * 255.0, 0, 255)

        def overlay(mask_stack: np.ndarray) -> np.ndarray:
            result = image_rgb.copy()
            for class_index, color in enumerate(colors):
                mask = mask_stack[class_index] > 0
                result[mask] = 0.55 * result[mask] + 0.45 * color
            return result.astype(np.uint8)

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, content, title in zip(
            axes,
            (image_rgb.astype(np.uint8), overlay(target.numpy()), overlay(prediction)),
            ("Fundus", "Ground truth", "Prediction"),
        ):
            axis.imshow(content)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(
            f"{image_id} | red=MA, magenta=HE, yellow=EX",
            fontsize=11,
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"{image_id}.png", dpi=130, bbox_inches="tight")
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.output_dir = prepare_output_dir(args.output_dir)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for lesion segmentation training")
    device = torch.device("cuda")
    amp_enabled = not args.no_amp

    splits = create_idrid_splits(
        args.dataset_dir,
        args.output_dir / "splits",
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    datasets = {
        "train": IDRiDLesionDataset(
            splits["train"], args.image_size, training=True, enhance=args.enhance
        ),
        "val": IDRiDLesionDataset(
            splits["val"], args.image_size, training=False, enhance=args.enhance
        ),
        "test": IDRiDLesionDataset(
            splits["test"], args.image_size, training=False, enhance=args.enhance
        ),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
    print(
        f"IDRiD splits: train={len(datasets['train'])}, val={len(datasets['val'])}, "
        f"test={len(datasets['test'])} (official test untouched until final evaluation)"
    )

    model = build_model(args).to(device)
    criterion = DiceFocalLoss(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
    )
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": decoder_parameters, "lr": args.decoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scaler = create_scaler(amp_enabled)

    best_val_dice = -1.0
    best_epoch = -1
    stale_epochs = 0
    history_path = args.output_dir / "history.jsonl"
    default_thresholds = [0.5] * len(LESION_NAMES)
    for epoch in range(args.epochs):
        encoder_frozen = epoch < args.freeze_epochs
        set_encoder_trainable(model, not encoder_frozen)
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            accum_steps=args.accum_steps,
            amp_enabled=amp_enabled,
            encoder_frozen=encoder_frozen,
        )
        val_loss, val_metrics, _, _ = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            thresholds=default_thresholds,
            amp_enabled=amp_enabled,
            collect=False,
            description="lesion-val",
        )
        current_dice = float(val_metrics["macro"]["dice"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "encoder_frozen": encoder_frozen,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "val_metrics": val_metrics,
            "test_evaluated": False,
        }
        print(json.dumps(record, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if not math.isfinite(current_dice):
            raise RuntimeError("Validation produced a non-finite macro Dice")
        if current_dice > best_val_dice:
            best_val_dice = current_dice
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "checkpoint-best.pth",
                model,
                args,
                epoch=epoch,
                best_val_dice=best_val_dice,
                thresholds=default_thresholds,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            args.output_dir / "checkpoint-last.pth",
            model,
            args,
            epoch=epoch,
            best_val_dice=best_val_dice,
            thresholds=default_thresholds,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        scheduler.step()
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without Dice improvement")
            break

    best_path = args.output_dir / "checkpoint-best.pth"
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"], strict=True)
    model.to(device)
    _, _, val_probabilities, val_targets = evaluate(
        model,
        loaders["val"],
        criterion,
        device,
        thresholds=default_thresholds,
        amp_enabled=amp_enabled,
        collect=True,
        description="threshold-calibration",
    )
    calibrated_thresholds = calibrate_dice_thresholds(val_probabilities, val_targets)
    test_loss, test_metrics, _, _ = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        thresholds=calibrated_thresholds,
        amp_enabled=amp_enabled,
        collect=False,
        description="lesion-test-once",
    )
    save_checkpoint(
        best_path,
        model,
        args,
        epoch=best_epoch,
        best_val_dice=best_val_dice,
        thresholds=calibrated_thresholds,
    )
    save_previews(
        model,
        datasets["test"],
        args.output_dir / "previews",
        device,
        thresholds=calibrated_thresholds,
        count=args.preview_count,
        amp_enabled=amp_enabled,
    )
    summary = {
        "task": "idrid_multilabel_lesion_segmentation",
        "lesions": list(LESION_NAMES),
        "architecture": args.architecture,
        "encoder_name": args.encoder_name,
        "encoder_weights": args.encoder_weights,
        "image_size": args.image_size,
        "split_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "best_epoch": best_epoch,
        "best_validation_macro_dice_at_0.5": best_val_dice,
        "calibrated_thresholds": dict(zip(LESION_NAMES, calibrated_thresholds)),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "test_evaluated_once_after_selection": True,
        "checkpoint": "checkpoint-best.pth",
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

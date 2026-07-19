import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adjust path to import correctly from root of dr-diagnostic-system
sys.path.append(os.fspath(Path(__file__).resolve().parents[2]))

from ai.grading.train import (
    FundusDataset,
    build_transforms,
)
from ai.semi_supervised.research_utils import (
    load_grading_checkpoint,
    load_split_frames,
)


class DataArgsNamespace:
    def __init__(self, dataset_dir, enhance, image_size):
        self.dataset_dir = dataset_dir
        self.images_dir = None
        self.labels_csv = None
        self.image_column = "id_code"
        self.label_column = "diagnosis"
        self.image_extension = ".png"
        self.enhance = enhance
        self.image_size = image_size


def main():
    parser = argparse.ArgumentParser(description="Analyze calibration and error rates by class at different confidence thresholds.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint-best.pth")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Path to split_dataset folder")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="Which split to analyze")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--enhance", action="store_true", help="Enhance fundus images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}...", flush=True)
    bundle = load_grading_checkpoint(args.checkpoint, device, require_ce=True)
    model, saved_args = bundle.model, bundle.saved_args
    model.eval()

    image_size = int(getattr(saved_args, "image_size", 224))
    _, eval_transform = build_transforms(image_size)

    # Load data
    print(f"Loading split frames from: {args.dataset_dir}...", flush=True)
    frames, _ = load_split_frames(args.dataset_dir)
    target_frame = frames[args.split]
    
    data_args = DataArgsNamespace(args.dataset_dir, args.enhance, image_size)
    dataset = FundusDataset(target_frame, data_args, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    # Structure to hold stats
    # For each class, and each threshold: {class_id: {threshold: {total_pred: int, incorrect: int}}}
    stats = {c: {t: {"total_pred": 0, "incorrect": 0} for t in thresholds} for c in range(5)}
    
    print("\n--- Image level predictions ---", flush=True)
    with torch.no_grad():
        for images, targets, image_ids in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)
            
            confidence, pred_labels = probabilities.max(dim=1)
            for img_id, true_lbl, pred_lbl, conf in zip(image_ids, targets, pred_labels, confidence):
                true_lbl = int(true_lbl)
                pred_lbl = int(pred_lbl)
                conf = float(conf)
                is_correct = (true_lbl == pred_lbl)
                
                status_str = "Correct" if is_correct else f"INCORRECT (True: {class_names[true_lbl]})"
                print(f"[{img_id}] Predicted: {class_names[pred_lbl]} ({conf*100:.2f}%) | {status_str}", flush=True)
                
                # Accumulate statistics
                for t in thresholds:
                    if conf >= t:
                        stats[pred_lbl][t]["total_pred"] += 1
                        if not is_correct:
                            stats[pred_lbl][t]["incorrect"] += 1

    print("\n" + "="*80)
    print("CALIBRATION & PSEUDO-LABEL ERROR RATE ANALYSIS REPORT")
    print("="*80)
    for c in range(5):
        print(f"\nClass {c} ({class_names[c]}):")
        print("-" * 50)
        for t in thresholds:
            cell = stats[c][t]
            tot = cell["total_pred"]
            inc = cell["incorrect"]
            err_rate = (inc / tot * 100) if tot > 0 else 0.0
            print(f"Threshold >= {t:.2f} ({t*100:.0f}%): Total Predicted = {tot:4d} | Incorrect = {inc:4d} (Error Rate: {err_rate:6.2f}%)")
    print("="*80)


if __name__ == "__main__":
    main()

# %% [markdown]
# # One-cell Colab: offline preprocessing fundus dataset to RGB crop, light Ben Graham, CLAHE LAB
#
# Copy toàn bộ cell này vào Google Colab và chạy. Output được tạo theo cấu trúc
# chuẩn để train trực tiếp:
#
# ```text
# /content/processed_fundus_dataset/
#   rgb_crop_512/train/0..4
#   rgb_crop_512/validation/0..4
#   rgb_crop_512/test/0..4
#   bengraham_light_512/train/0..4
#   bengraham_light_512/validation/0..4
#   bengraham_light_512/test/0..4
#   clahe_lab_512/train/0..4
#   clahe_lab_512/validation/0..4
#   clahe_lab_512/test/0..4
# ```
#
# Khuyến nghị train:
#
# ```bash
# python -m ai.grading.train_pretrained_baselines \
#   --dataset-dir /content/processed_fundus_dataset/rgb_crop_512 \
#   --architecture efficientnet_b3 \
#   --preprocessing preprocessed \
#   --image-size 384
# ```

# %%capture
!pip -q install opencv-python-headless pillow numpy pandas matplotlib tqdm scikit-learn

# %%
from __future__ import annotations

import csv
import json
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print("Drive mount skipped:", exc)


# =========================
# CONFIG - chỉnh ở đây
# =========================

# Nếu bạn đã có zip dataset trên Drive, đặt đường dẫn này. Zip nên chứa
# train/validation/test/0..4 hoặc split_dataset/train/validation/test/0..4.
DATASET_ZIP_ON_DRIVE = Path("/content/drive/MyDrive/split_dataset.zip")

# Nếu DATASET_ZIP_ON_DRIVE không tồn tại, script sẽ dùng RAW_DATA_DIR nếu đã có ảnh.
RAW_DATA_DIR = Path("/content/raw_fundus_dataset")

# Output local trên /content để xử lý nhanh, cuối cell sẽ nén/copy sang Drive.
OUTPUT_ROOT = Path("/content/processed_fundus_dataset")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/processed_fundus_dataset")

TARGET_SIZE = 512
OUTPUT_FORMAT = "jpg"
JPEG_QUALITY = 95
NUM_WORKERS = 8
RANDOM_STATE = 42

# Các variant sẽ được tạo. `rgb_crop_512` thường là baseline tốt nhất cho
# EfficientNet/ConvNeXt pretrained. `bengraham_light_512` làm nổi mạch máu/tổn
# thương nhưng giữ màu tốt hơn công thức 4/-4/128. `clahe_lab_512` tăng tương
# phản nhẹ trên kênh L của LAB, giữ màu tốt hơn CLAHE từng kênh RGB.
CREATE_RGB_CROP = True
CREATE_BENGRAHAM_LIGHT = True
CREATE_CLAHE_LAB = True

# Ben Graham nhẹ: chạy sau crop+resize nên sigma ổn định. Nếu ảnh vẫn bạc màu,
# giảm BEN_GRAHAM_ALPHA về 1.2 hoặc tăng BEN_GRAHAM_BETA về -0.2.
BEN_GRAHAM_LIGHT_ALPHA = 1.5
BEN_GRAHAM_LIGHT_BETA = -0.5
BEN_GRAHAM_LIGHT_GAMMA = 0.0
BEN_GRAHAM_LIGHT_SIGMA = 10.0
BEN_GRAHAM_LIGHT_BLEND_ORIGINAL = 0.35

# CLAHE nhẹ. Nếu ảnh bị gắt/noisy, giảm clipLimit xuống 1.2-1.5.
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)

BLACK_THRESHOLD = 10
CROP_MARGIN_RATIO = 0.03
MIN_FUNDUS_AREA_RATIO = 0.20
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "validation": ("validation", "valid", "val"),
    "test": ("test", "testing"),
}

RESUME = True
OVERWRITE_EXISTING = False
CREATE_OUTPUT_ZIP = True


# =========================
# Helpers
# =========================

def extract_raw_dataset() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    has_images = any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in RAW_DATA_DIR.rglob("*")
    )
    if has_images:
        print(f"RAW_DATA_DIR already has images: {RAW_DATA_DIR}")
        return
    if not DATASET_ZIP_ON_DRIVE.exists():
        raise FileNotFoundError(
            f"Không thấy {DATASET_ZIP_ON_DRIVE}. Hãy upload zip dataset lên Drive "
            "hoặc giải nén sẵn vào RAW_DATA_DIR."
        )
    print(f"Extracting {DATASET_ZIP_ON_DRIVE} -> {RAW_DATA_DIR}")
    with zipfile.ZipFile(DATASET_ZIP_ON_DRIVE, "r") as archive:
        archive.extractall(RAW_DATA_DIR)


def find_split_root(root: Path) -> Path:
    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    for candidate in candidates:
        children = {child.name.lower(): child for child in candidate.iterdir() if child.is_dir()}
        found = {}
        for canonical, aliases in SPLIT_ALIASES.items():
            match = next((children[alias] for alias in aliases if alias in children), None)
            if match is not None:
                found[canonical] = match
        if set(found) == {"train", "validation", "test"}:
            print("Using split root:", candidate)
            return candidate
    raise FileNotFoundError(
        f"Không tìm thấy cấu trúc train/validation/test dưới {root}. "
        "Dataset cần có split sẵn để tránh chia lại và tránh leakage."
    )


def list_tasks(split_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for canonical_split, aliases in SPLIT_ALIASES.items():
        split_dir = next(
            split_root / alias
            for alias in aliases
            if (split_root / alias).is_dir()
        )
        for label_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            label = label_dir.name.strip()
            if label not in {"0", "1", "2", "3", "4"}:
                continue
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    relative_stem = image_path.relative_to(split_dir).with_suffix("")
                    safe_name = "_".join(relative_stem.parts) + f".{OUTPUT_FORMAT}"
                    tasks.append(
                        {
                            "src": str(image_path),
                            "split": canonical_split,
                            "label": label,
                            "name": safe_name,
                        }
                    )
    if not tasks:
        raise FileNotFoundError(f"Không thấy ảnh hợp lệ dưới {split_root}")
    return tasks


def read_rgb(path: str | os.PathLike[str]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def black_ratio_rgb(image: np.ndarray, threshold: int = BLACK_THRESHOLD) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float((gray <= threshold).mean())


def crop_fundus_rgb(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    info = {
        "crop_status": "fallback",
        "crop_x1": 0,
        "crop_y1": 0,
        "crop_x2": w,
        "crop_y2": h,
        "cropped_width": w,
        "cropped_height": h,
        "black_ratio_after_crop": black_ratio_rgb(image),
    }
    if not contours:
        return image.copy(), info

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < MIN_FUNDUS_AREA_RATIO * h * w:
        return image.copy(), info

    x, y, bw, bh = cv2.boundingRect(contour)
    margin = int(round(max(bw, bh) * CROP_MARGIN_RATIO))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)
    cropped = image[y1:y2, x1:x2].copy()
    info.update(
        {
            "crop_status": "ok",
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "cropped_width": x2 - x1,
            "cropped_height": y2 - y1,
            "black_ratio_after_crop": black_ratio_rgb(cropped),
        }
    )
    return cropped, info


def resize_with_padding_rgb(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def clahe_lab_rgb(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_l, a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )
    return enhanced.astype(np.uint8)


def ben_graham_light_rgb(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=BEN_GRAHAM_LIGHT_SIGMA,
        sigmaY=BEN_GRAHAM_LIGHT_SIGMA,
    )
    enhanced = cv2.addWeighted(
        img,
        BEN_GRAHAM_LIGHT_ALPHA,
        blurred,
        BEN_GRAHAM_LIGHT_BETA,
        BEN_GRAHAM_LIGHT_GAMMA,
    )
    if BEN_GRAHAM_LIGHT_BLEND_ORIGINAL > 0:
        enhanced = (
            (1.0 - BEN_GRAHAM_LIGHT_BLEND_ORIGINAL) * enhanced
            + BEN_GRAHAM_LIGHT_BLEND_ORIGINAL * img
        )
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    enhanced[gray <= BLACK_THRESHOLD] = 0
    return enhanced


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    if OUTPUT_FORMAT.lower() in {"jpg", "jpeg"}:
        pil_image.save(path, quality=JPEG_QUALITY, subsampling=0, optimize=True)
    else:
        pil_image.save(path)


def output_path(variant: str, task: dict[str, Any]) -> Path:
    return OUTPUT_ROOT / variant / task["split"] / task["label"] / task["name"]


def process_one(task: dict[str, Any]) -> dict[str, Any]:
    rgb_path = output_path("rgb_crop_512", task)
    bg_light_path = output_path("bengraham_light_512", task)
    clahe_path = output_path("clahe_lab_512", task)
    expected_paths = []
    if CREATE_RGB_CROP:
        expected_paths.append(rgb_path)
    if CREATE_BENGRAHAM_LIGHT:
        expected_paths.append(bg_light_path)
    if CREATE_CLAHE_LAB:
        expected_paths.append(clahe_path)
    if RESUME and not OVERWRITE_EXISTING and expected_paths and all(path.exists() for path in expected_paths):
        return {
            **task,
            "status": "skipped_existing",
            "rgb_crop_path": str(rgb_path) if CREATE_RGB_CROP else "",
            "bengraham_light_path": str(bg_light_path) if CREATE_BENGRAHAM_LIGHT else "",
            "clahe_lab_path": str(clahe_path) if CREATE_CLAHE_LAB else "",
            "error": "",
        }

    try:
        image = read_rgb(task["src"])
        original_h, original_w = image.shape[:2]
        cropped, crop_info = crop_fundus_rgb(image)
        rgb_crop = resize_with_padding_rgb(cropped, TARGET_SIZE)

        if CREATE_RGB_CROP:
            save_rgb(rgb_crop, rgb_path)
        if CREATE_BENGRAHAM_LIGHT:
            save_rgb(ben_graham_light_rgb(rgb_crop), bg_light_path)
        if CREATE_CLAHE_LAB:
            # CLAHE after crop+resize keeps its strength stable across datasets.
            save_rgb(clahe_lab_rgb(rgb_crop), clahe_path)

        return {
            **task,
            "status": "success",
            "original_width": original_w,
            "original_height": original_h,
            **crop_info,
            "rgb_crop_path": str(rgb_path) if CREATE_RGB_CROP else "",
            "bengraham_light_path": str(bg_light_path) if CREATE_BENGRAHAM_LIGHT else "",
            "clahe_lab_path": str(clahe_path) if CREATE_CLAHE_LAB else "",
            "error": "",
        }
    except Exception as exc:
        return {
            **task,
            "status": "error",
            "rgb_crop_path": str(rgb_path) if CREATE_RGB_CROP else "",
            "bengraham_light_path": str(bg_light_path) if CREATE_BENGRAHAM_LIGHT else "",
            "clahe_lab_path": str(clahe_path) if CREATE_CLAHE_LAB else "",
            "error": repr(exc),
        }


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def summarize_outputs(manifest: pd.DataFrame) -> None:
    print("\nStatus:")
    display(manifest["status"].value_counts())
    print("\nSplit/class:")
    display(pd.crosstab(manifest["split"], manifest["label"]))
    failed = manifest[manifest["status"] == "error"]
    if not failed.empty:
        print("\nErrors:")
        display(failed[["src", "error"]].head(20))


def show_samples(manifest: pd.DataFrame, n: int = 6) -> None:
    sample = manifest[manifest["status"].isin(["success", "skipped_existing"])].sample(
        min(n, len(manifest)),
        random_state=RANDOM_STATE,
    )
    for row in sample.itertuples(index=False):
        images = [read_rgb(row.src)]
        titles = ["Original"]
        if CREATE_RGB_CROP and row.rgb_crop_path:
            images.append(read_rgb(row.rgb_crop_path))
            titles.append("RGB crop")
        if CREATE_BENGRAHAM_LIGHT and row.bengraham_light_path:
            images.append(read_rgb(row.bengraham_light_path))
            titles.append("Ben Graham light")
        if CREATE_CLAHE_LAB and row.clahe_lab_path:
            images.append(read_rgb(row.clahe_lab_path))
            titles.append("CLAHE LAB")

        fig, axes = plt.subplots(1, len(images), figsize=(4.5 * len(images), 4))
        if len(images) == 1:
            axes = [axes]
        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"split={row.split} | label={row.label} | {Path(row.src).name}")
        plt.show()


def zip_variant(variant: str) -> Path | None:
    variant_dir = OUTPUT_ROOT / variant
    if not variant_dir.exists():
        return None
    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_base = DRIVE_OUTPUT_DIR / variant
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=variant_dir)
    return Path(zip_path)


# =========================
# Run
# =========================

for directory in [OUTPUT_ROOT, DRIVE_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

extract_raw_dataset()
split_root = find_split_root(RAW_DATA_DIR)
tasks = list_tasks(split_root)
print(f"Total images: {len(tasks):,}")
print(pd.DataFrame(tasks).groupby(["split", "label"]).size().unstack(fill_value=0))

manifest_path = DRIVE_OUTPUT_DIR / "preprocessing_manifest.csv"
rows: list[dict[str, Any]] = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_one, task) for task in tasks]
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"), start=1):
        rows.append(future.result())
        if i % 1000 == 0:
            write_manifest(rows, manifest_path)

write_manifest(rows, manifest_path)
manifest_df = pd.DataFrame(rows)
summarize_outputs(manifest_df)

config = {
    "target_size": TARGET_SIZE,
    "output_format": OUTPUT_FORMAT,
    "jpeg_quality": JPEG_QUALITY,
    "clahe_clip_limit": CLAHE_CLIP_LIMIT,
    "clahe_tile_grid_size": CLAHE_TILE_GRID_SIZE,
    "ben_graham_light_alpha": BEN_GRAHAM_LIGHT_ALPHA,
    "ben_graham_light_beta": BEN_GRAHAM_LIGHT_BETA,
    "ben_graham_light_gamma": BEN_GRAHAM_LIGHT_GAMMA,
    "ben_graham_light_sigma": BEN_GRAHAM_LIGHT_SIGMA,
    "ben_graham_light_blend_original": BEN_GRAHAM_LIGHT_BLEND_ORIGINAL,
    "black_threshold": BLACK_THRESHOLD,
    "crop_margin_ratio": CROP_MARGIN_RATIO,
    "min_fundus_area_ratio": MIN_FUNDUS_AREA_RATIO,
    "create_rgb_crop": CREATE_RGB_CROP,
    "create_bengraham_light": CREATE_BENGRAHAM_LIGHT,
    "create_clahe_lab": CREATE_CLAHE_LAB,
    "output_root": str(OUTPUT_ROOT),
    "drive_output_dir": str(DRIVE_OUTPUT_DIR),
}
(DRIVE_OUTPUT_DIR / "preprocessing_config.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

show_samples(manifest_df, n=6)

if CREATE_OUTPUT_ZIP:
    created = []
    if CREATE_RGB_CROP:
        created.append(zip_variant("rgb_crop_512"))
    if CREATE_BENGRAHAM_LIGHT:
        created.append(zip_variant("bengraham_light_512"))
    if CREATE_CLAHE_LAB:
        created.append(zip_variant("clahe_lab_512"))
    print("\nCreated zip files:")
    for path in created:
        if path is not None:
            print(" -", path)

print("\nDone.")
print("Train RGB crop with:")
print(
    "python -m ai.grading.train_pretrained_baselines "
    "--dataset-dir /content/processed_fundus_dataset/rgb_crop_512 "
    "--architecture efficientnet_b3 --preprocessing preprocessed --image-size 384"
)
print("Train light Ben Graham with:")
print(
    "python -m ai.grading.train_pretrained_baselines "
    "--dataset-dir /content/processed_fundus_dataset/bengraham_light_512 "
    "--architecture efficientnet_b3 --preprocessing preprocessed --image-size 384"
)
print("Train CLAHE LAB with:")
print(
    "python -m ai.grading.train_pretrained_baselines "
    "--dataset-dir /content/processed_fundus_dataset/clahe_lab_512 "
    "--architecture convnext_tiny --preprocessing preprocessed --image-size 384"
)

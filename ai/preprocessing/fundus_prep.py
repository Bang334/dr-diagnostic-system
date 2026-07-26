"""Shared fundus preprocessing for training and inference.

The default path deliberately preserves RGB colour.  Green-channel-only
preprocessing is useful for some lesion-segmentation experiments, but it throws
away colour information that a pretrained grading backbone expects.
"""

from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from tqdm import tqdm


BLACK_THRESHOLD = 10
CROP_MARGIN_RATIO = 0.03
MIN_FUNDUS_AREA_RATIO = 0.20


def crop_image_from_gray(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """Remove the black camera border while preserving every colour channel."""
    if image is None or image.size == 0:
        raise ValueError("Fundus image is empty")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Expected HxW or HxWx3 image, received {image.shape}")

    mask = gray > tol
    if not mask.any():
        return image.copy()

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def crop_fundus_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Crop the retinal field using the contour logic used by rgb_crop_512."""
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Fundus image is empty")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, received {image_rgb.shape}")

    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_rgb.copy()

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_FUNDUS_AREA_RATIO * height * width:
        return image_rgb.copy()

    x, y, crop_width, crop_height = cv2.boundingRect(contour)
    margin = int(round(max(crop_width, crop_height) * CROP_MARGIN_RATIO))
    return image_rgb[
        max(0, y - margin) : min(height, y + crop_height + margin),
        max(0, x - margin) : min(width, x + crop_width + margin),
    ].copy()


def resize_with_padding_rgb(
    image_rgb: np.ndarray, target_size: int = 512
) -> np.ndarray:
    """Resize an RGB image without distortion and pad it to a square."""
    height, width = image_rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Fundus crop is empty")
    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def preprocess_rgb_crop_512_from_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Reproduce the teacher model's colour-preserving rgb_crop_512 input."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Fundus image is empty")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return resize_with_padding_rgb(crop_fundus_rgb(image_rgb), 512)


def ben_graham_enhance(image_bgr: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Correct uneven illumination without collapsing the image to grayscale."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image_bgr, 4.0, blurred, -4.0, 128.0)


def preprocess_fundus_array(
    image_bgr: np.ndarray,
    img_size: int | Tuple[int, int] | None = None,
    *,
    enhance: bool = False,
    crop_tolerance: int = 7,
) -> np.ndarray:
    """Crop a BGR fundus image, optionally enhance it, and resize it.

    The return value remains BGR so OpenCV callers do not silently swap colour
    channels.  Conversion to RGB belongs at the model boundary.
    """
    image = crop_image_from_gray(image_bgr, tol=crop_tolerance)
    if enhance:
        image = ben_graham_enhance(image)

    if img_size is not None:
        size = (img_size, img_size) if isinstance(img_size, int) else img_size
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def preprocess_fundus_image(
    image_path: str | os.PathLike[str],
    img_size: int = 512,
    *,
    enhance: bool = False,
) -> np.ndarray | None:
    """Read and preprocess one fundus image while preserving RGB information."""
    image = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return preprocess_fundus_array(image, img_size, enhance=enhance)


def _process_single_file(task: tuple[str, str, int, bool]) -> bool:
    src_path, dst_path, img_size, enhance = task
    if os.path.exists(dst_path):
        return True

    image = preprocess_fundus_image(src_path, img_size, enhance=enhance)
    if image is None:
        return False
    return bool(cv2.imwrite(dst_path, image))


def _find_images(src_dir: str, pattern: str) -> Iterable[str]:
    if pattern != "auto":
        return glob.glob(os.path.join(src_dir, pattern))
    paths: list[str] = []
    for extension in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(src_dir, extension)))
    return sorted(paths)


def batch_process(
    src_dir: str,
    dst_dir: str,
    img_size: int = 512,
    ext: str = "auto",
    num_workers: int = 4,
    *,
    enhance: bool = False,
) -> None:
    """Preprocess a directory; existing output files make the job resumable."""
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    image_paths = list(_find_images(src_dir, ext))
    if not image_paths:
        print(f"[!] No images found in {src_dir!r} with pattern {ext!r}")
        return

    tasks = [
        (path, os.path.join(dst_dir, os.path.basename(path)), img_size, enhance)
        for path in image_paths
    ]
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for succeeded in tqdm(
            executor.map(_process_single_file, tasks),
            total=len(tasks),
            desc="Preprocessing",
        ):
            success_count += int(succeeded)
    print(f"[v] Processed {success_count}/{len(tasks)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess retinal fundus images")
    parser.add_argument("--src", required=True, help="Directory containing raw images")
    parser.add_argument("--dst", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--ext",
        default="auto",
        help="Glob such as '*.png'; 'auto' reads PNG/JPG/JPEG",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply colour-preserving Ben Graham illumination correction",
    )
    args = parser.parse_args()
    batch_process(
        args.src,
        args.dst,
        args.size,
        args.ext,
        args.workers,
        enhance=args.enhance,
    )

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

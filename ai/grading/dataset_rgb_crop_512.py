# # Notebook tiền xử lý offline ảnh võng mạc cho Diabetic Retinopathy
#
# Notebook/script này được thiết kế để chạy trực tiếp trên Google Colab. Bạn có thể upload file `.py` này lên Colab hoặc copy từng cell. Pipeline tạo 2 phiên bản dataset:
#
# - `rgb_crop_512`: ảnh RGB đã kiểm tra lỗi, crop vùng fundus, resize và padding 512x512.
# - `rgb_bengraham_512`: giống baseline nhưng thêm Ben Graham preprocessing.
#
# Dataset gốc đã tải sẵn trên Google Drive thì chỉ cần chỉnh `PROJECT_DIR` và `RAW_DATA_DIR` ở cell cấu hình. Phần Kaggle API là tùy chọn và sẽ không tải lại nếu thư mục raw đã tồn tại.

# ## 1. Cài đặt thư viện

!pip -q install opencv-python-headless Pillow numpy pandas matplotlib scikit-learn tqdm imagehash kaggle

# ## 2. Mount Google Drive và cấu hình tập trung

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import os
import json
import math
import shutil
import hashlib
import logging
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

PROJECT_DIR = Path("/content/drive/MyDrive/DR_Project_fresh_run")
# Drive chỉ dùng để lưu zip/manifest cuối mỗi chunk. Raw và ảnh xử lý chạy trên /content cho nhanh.
DRIVE_OUTPUT_DIR = PROJECT_DIR / "processed_dataset"

LOCAL_PROJECT_DIR = Path("/content/DR_Project_fresh_run")
RAW_DATA_DIR = LOCAL_PROJECT_DIR / "raw_dataset"
PROCESSED_DATA_DIR = LOCAL_PROJECT_DIR / "processed_dataset"
REPORT_DIR = DRIVE_OUTPUT_DIR / "reports"
LOG_DIR = DRIVE_OUTPUT_DIR / "logs"

KAGGLE_DATASET = "sehastrajits/fundus-aptosddridirdeyepacsmessidor"

TARGET_SIZE = 512
OUTPUT_FORMAT = "jpg"
JPEG_QUALITY = 95

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

BLACK_THRESHOLD = 10
CROP_MARGIN_RATIO = 0.03
MIN_FUNDUS_AREA_RATIO = 0.20
MIN_IMAGE_SIDE = 64
MAX_BLACK_RATIO_WARN = 0.85

BEN_GRAHAM_ALPHA = 4.0
BEN_GRAHAM_BETA = -4.0
BEN_GRAHAM_GAMMA = 128
BEN_GRAHAM_SIGMA_RATIO = 1 / 30

ENABLE_CLAHE_DATASET = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ENABLE_PERCEPTUAL_DUPLICATE_CHECK = False
PHASH_DISTANCE_THRESHOLD = 4

NUM_WORKERS = 2
BATCH_SIZE = 32
RESUME_PROCESSING = True
CHECKPOINT_INTERVAL = 500
OVERWRITE_EXISTING = False

DRY_RUN = False
DRY_RUN_IMAGES = 100

# Xử lý theo từng chunk để phù hợp Colab Free.
# Mỗi chunk xử lý tối đa 10k ảnh, ghi manifest riêng và có thể nén ngay vào Google Drive.
PROCESS_IN_CHUNKS = True
CHUNK_SIZE = 10_000
ZIP_EACH_CHUNK = True
DELETE_UNCOMPRESSED_CHUNK_AFTER_ZIP = True

# Colab Free có ổ /content nhanh hơn Google Drive nhưng dung lượng giới hạn.
# Pipeline mặc định giải nén và xử lý trên /content, sau mỗi chunk sẽ nén và copy zip sang Drive.
USE_LOCAL_RUNTIME = True
LOCAL_RUNTIME_DIR = LOCAL_PROJECT_DIR
MIN_FREE_SPACE_GB = 10

# File zip dataset bạn đã tải sẵn trên Google Drive.
DATASET_ZIP_ON_DRIVE = Path("/content/drive/MyDrive/split_dataset.zip")
# Fresh run cho tài khoản Colab mới: dùng thư mục project/output mới để không dính manifest cũ.
# Vẫn giữ RESUME_PROCESSING=True để nếu Colab bị ngắt giữa chừng, lần chạy lại có thể tiếp tục các chunk đã hoàn tất.
FRESH_RUN_NAME = "DR_Project_fresh_run"

# Nếu tự động đoán sai metadata, sửa 4 biến này sau khi chạy cell khảo sát.
METADATA_CSV = None
IMAGE_COLUMN = None
LABEL_COLUMN = None
PATIENT_COLUMN = None
SOURCE_COLUMN = None
EYE_COLUMN = None
SPLIT_COLUMN = None
USE_EXISTING_SPLIT_IF_AVAILABLE = True

for d in [PROJECT_DIR, DRIVE_OUTPUT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocessing.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("dr_preprocess")

CONFIG = {
    "target_size": TARGET_SIZE,
    "output_format": OUTPUT_FORMAT,
    "jpeg_quality": JPEG_QUALITY,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "random_state": RANDOM_STATE,
    "black_threshold": BLACK_THRESHOLD,
    "crop_margin_ratio": CROP_MARGIN_RATIO,
    "min_fundus_area_ratio": MIN_FUNDUS_AREA_RATIO,
    "ben_graham_alpha": BEN_GRAHAM_ALPHA,
    "ben_graham_beta": BEN_GRAHAM_BETA,
    "ben_graham_gamma": BEN_GRAHAM_GAMMA,
    "ben_graham_sigma_ratio": BEN_GRAHAM_SIGMA_RATIO,
    "enable_clahe_dataset": ENABLE_CLAHE_DATASET,
    "dry_run": DRY_RUN,
    "dry_run_images": DRY_RUN_IMAGES,
    "process_in_chunks": PROCESS_IN_CHUNKS,
    "chunk_size": CHUNK_SIZE,
    "zip_each_chunk": ZIP_EACH_CHUNK,
    "delete_uncompressed_chunk_after_zip": DELETE_UNCOMPRESSED_CHUNK_AFTER_ZIP,
    "use_existing_split_if_available": USE_EXISTING_SPLIT_IF_AVAILABLE,
    "use_local_runtime": USE_LOCAL_RUNTIME,
    "min_free_space_gb": MIN_FREE_SPACE_GB,
}
(DRIVE_OUTPUT_DIR / "preprocessing_config.json").write_text(
    json.dumps(CONFIG, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("PROJECT_DIR:", PROJECT_DIR)
print("DRIVE_OUTPUT_DIR:", DRIVE_OUTPUT_DIR)
print("LOCAL_PROJECT_DIR:", LOCAL_PROJECT_DIR)
print("RAW_DATA_DIR:", RAW_DATA_DIR)
print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)

# ## 2.1. Kiểm tra dung lượng để tránh Colab Free bị đầy ổ
#
# Colab Free có dung lượng runtime `/content` giới hạn nhưng tốc độ đọc/ghi nhanh hơn Google Drive rất nhiều. Pipeline này giải nén raw dataset vào `/content`, xử lý output tạm ở `/content`, rồi nén từng chunk và copy zip/manifest sang Google Drive.

def bytes_to_gb(n: int) -> float:
    return n / (1024 ** 3)

def disk_report(path: Path) -> Dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": round(bytes_to_gb(usage.total), 2),
        "used_gb": round(bytes_to_gb(usage.used), 2),
        "free_gb": round(bytes_to_gb(usage.free), 2),
    }

def folder_size_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(bytes_to_gb(total), 2)

def print_storage_report() -> None:
    print("/content:", disk_report(Path("/content")))
    print("/content/drive:", disk_report(Path("/content/drive")))
    print("RAW_DATA_DIR size GB:", folder_size_gb(RAW_DATA_DIR))
    print("PROCESSED_DATA_DIR size GB:", folder_size_gb(PROCESSED_DATA_DIR))
    print("DRIVE_OUTPUT_DIR size GB:", folder_size_gb(DRIVE_OUTPUT_DIR))
    free_content = disk_report(Path("/content"))["free_gb"]
    if free_content < MIN_FREE_SPACE_GB:
        print(f"CẢNH BÁO: /content chỉ còn {free_content} GB. Có thể không đủ để giải nén/xử lý local.")

print_storage_report()

# ## 3. Chuẩn bị raw dataset trên `/content`
#
# Để tăng tốc, dataset sẽ được giải nén vào `/content/DR_Project/raw_dataset`. Nếu bạn đã có file zip trên Google Drive, đặt `DATASET_ZIP_ON_DRIVE` ở cell cấu hình. Nếu chưa có zip, có thể dùng Kaggle API để tải thẳng về `/content`.

def raw_dataset_has_images() -> bool:
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return any(p.suffix.lower() in image_exts for p in RAW_DATA_DIR.rglob("*"))

def extract_zip_to_raw(zip_path: Path) -> None:
    """Giải nén zip dataset vào RAW_DATA_DIR trên /content."""
    import zipfile
    if not zip_path.exists():
        raise FileNotFoundError(f"Không thấy file zip: {zip_path}")
    print_storage_report()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Giải nén %s vào %s", zip_path, RAW_DATA_DIR)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DATA_DIR)
    print_storage_report()

def prepare_raw_dataset_local() -> None:
    """Chuẩn bị dataset raw trên /content để xử lý nhanh."""
    if raw_dataset_has_images():
        logger.info("RAW_DATA_DIR trên /content đã có ảnh, bỏ qua giải nén/tải lại.")
        return

    if DATASET_ZIP_ON_DRIVE is not None:
        extract_zip_to_raw(Path(DATASET_ZIP_ON_DRIVE))
        return

    maybe_download_kaggle_dataset()

def maybe_download_kaggle_dataset() -> None:
    """Download Kaggle dataset thẳng vào /content rồi giải nén vào RAW_DATA_DIR."""
    from google.colab import files
    print("RAW_DATA_DIR chưa có ảnh. Hãy upload kaggle.json để tải dataset vào /content.")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise FileNotFoundError("Không thấy kaggle.json trong file upload.")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    shutil.copy("kaggle.json", kaggle_dir / "kaggle.json")
    os.chmod(kaggle_dir / "kaggle.json", 0o600)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_dir = LOCAL_PROJECT_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    !kaggle datasets download -d {KAGGLE_DATASET} -p {str(download_dir)}

    zip_files = sorted(download_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError("Kaggle download xong nhưng không tìm thấy file zip trong /content.")

    extract_zip_to_raw(zip_files[0])
    logger.info("Hoàn tất tải và giải nén dataset vào /content.")

# Chạy tự động để chuẩn bị raw dataset local trước khi khảo sát.
prepare_raw_dataset_local()

# ## 4. Khảo sát cấu trúc dataset
#
# Cell này không giả định cứng tên thư mục hoặc tên CSV. Nó liệt kê ảnh, CSV và gợi ý các cột metadata có khả năng chứa tên ảnh, nhãn, patient/source.

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def list_dataset_files(root: Path) -> Tuple[List[Path], List[Path]]:
    images = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    csvs = sorted([p for p in root.rglob("*.csv") if p.is_file()])
    return images, csvs

def inspect_dataset(root: Path) -> Dict[str, Any]:
    images, csvs = list_dataset_files(root)
    print(f"Số ảnh tìm thấy: {len(images):,}")
    print(f"Số CSV tìm thấy: {len(csvs):,}")
    print("\nMột số thư mục:")
    for p in sorted([x for x in root.rglob("*") if x.is_dir()])[:80]:
        print(" -", p.relative_to(root))
    print("\nCSV và cột:")
    for csv in csvs:
        try:
            df = pd.read_csv(csv, nrows=5)
            print(f"\n{csv.relative_to(root)}")
            print(list(df.columns))
            display(df.head())
        except Exception as e:
            print("Không đọc được", csv, e)
    return {"images": images, "csvs": csvs}

survey = inspect_dataset(RAW_DATA_DIR)

def guess_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    low = {c: str(c).lower().strip() for c in cols}

    def first_match(keys: List[str]) -> Optional[str]:
        for key in keys:
            for c, lc in low.items():
                if key == lc or key in lc:
                    return c
        return None

    return {
        "image": first_match(["image", "image_id", "img", "filename", "file_name", "path", "id_code", "name"]),
        "label": first_match(["label", "diagnosis", "grade", "level", "class", "dr"]),
        "patient": first_match(["patient_id", "patient", "subject", "pid"]),
        "source": first_match(["source", "dataset", "origin"]),
        "eye": first_match(["eye", "laterality", "left_right", "side"]),
    }

def choose_metadata_csv(csvs: List[Path]) -> Optional[Path]:
    best = None
    best_score = -1
    for csv in csvs:
        try:
            df = pd.read_csv(csv, nrows=100)
            g = guess_columns(df)
            score = int(g["image"] is not None) + int(g["label"] is not None) * 2
            if score > best_score:
                best, best_score = csv, score
        except Exception:
            pass
    return best

def create_metadata_from_split_folders(root: Path) -> Optional[Path]:
    """Tạo metadata CSV khi dataset đã chia dạng train/val/test/class."""
    candidate_roots = [root] + [p for p in root.iterdir() if p.is_dir()]
    best_base = None
    best_count = 0
    for base in candidate_roots:
        count = 0
        for split in ["train", "val", "test"]:
            split_dir = base / split
            if not split_dir.exists():
                continue
            for label_dir in split_dir.iterdir():
                if label_dir.is_dir() and label_dir.name.isdigit():
                    count += sum(1 for p in label_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if count > best_count:
            best_base = base
            best_count = count

    if best_base is None or best_count == 0:
        return None

    rows = []
    for split in ["train", "val", "test"]:
        split_dir = best_base / split
        if not split_dir.exists():
            continue
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir() or not label_dir.name.isdigit():
                continue
            label = int(label_dir.name)
            if label not in range(5):
                continue
            for p in label_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTS:
                    rows.append({
                        "image": str(p),
                        "label": label,
                        "source": "unknown",
                        "patient_id": "",
                        "existing_split": split,
                    })

    meta = pd.DataFrame(rows)
    out = root / "auto_metadata_from_folders.csv"
    meta.to_csv(out, index=False)
    logger.info("Không thấy CSV metadata. Đã tự tạo %s từ folder split với %s ảnh.", out, len(meta))
    return out

if METADATA_CSV is None:
    METADATA_CSV = choose_metadata_csv(survey["csvs"])

if METADATA_CSV is None:
    METADATA_CSV = create_metadata_from_split_folders(RAW_DATA_DIR)

if METADATA_CSV is None:
    raise FileNotFoundError(
        "Không tìm thấy CSV metadata và cũng không thấy cấu trúc train/val/test/0..4. "
        "Hãy kiểm tra RAW_DATA_DIR hoặc đặt METADATA_CSV thủ công."
    )

meta_preview = pd.read_csv(METADATA_CSV, nrows=100)
guessed = guess_columns(meta_preview)
IMAGE_COLUMN = IMAGE_COLUMN or guessed["image"]
LABEL_COLUMN = LABEL_COLUMN or guessed["label"]
PATIENT_COLUMN = PATIENT_COLUMN or guessed["patient"]
SOURCE_COLUMN = SOURCE_COLUMN or guessed["source"]
EYE_COLUMN = EYE_COLUMN or guessed["eye"]
SPLIT_COLUMN = SPLIT_COLUMN or ("existing_split" if "existing_split" in meta_preview.columns else None)

print("METADATA_CSV:", METADATA_CSV)
print("IMAGE_COLUMN:", IMAGE_COLUMN)
print("LABEL_COLUMN:", LABEL_COLUMN)
print("PATIENT_COLUMN:", PATIENT_COLUMN)
print("SOURCE_COLUMN:", SOURCE_COLUMN)
print("EYE_COLUMN:", EYE_COLUMN)
print("SPLIT_COLUMN:", SPLIT_COLUMN)

if IMAGE_COLUMN is None or LABEL_COLUMN is None:
    raise ValueError("Không đoán chắc được IMAGE_COLUMN hoặc LABEL_COLUMN. Hãy sửa biến cấu hình rồi chạy lại.")

# ## 5. Đọc metadata, ánh xạ đường dẫn ảnh và chuẩn hóa nhãn

LABEL_MAP = {
    "0": 0, "no dr": 0, "normal": 0, "none": 0,
    "1": 1, "mild": 1,
    "2": 2, "moderate": 2,
    "3": 3, "severe": 3,
    "4": 4, "proliferative dr": 4, "pdr": 4, "proliferative": 4,
}

def normalize_label(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)) and int(x) in range(5):
        return int(x)
    if isinstance(x, float) and x.is_integer() and int(x) in range(5):
        return int(x)
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    try:
        v = int(float(s))
        return v if v in range(5) else None
    except Exception:
        return None

def build_image_index(images: List[Path]) -> Dict[str, Path]:
    index = {}
    for p in images:
        names = {p.name.lower(), p.stem.lower(), str(p.relative_to(RAW_DATA_DIR)).replace("\\", "/").lower()}
        for name in names:
            index.setdefault(name, p)
    return index

def resolve_image_path(value: Any, index: Dict[str, Path]) -> Optional[Path]:
    if pd.isna(value):
        return None
    s = str(value).strip().replace("\\", "/")
    candidates = [s.lower(), Path(s).name.lower(), Path(s).stem.lower()]
    for c in candidates:
        if c in index:
            return index[c]
    for ext in IMAGE_EXTS:
        key = (s + ext).lower()
        if key in index:
            return index[key]
        key = (Path(s).name + ext).lower()
        if key in index:
            return index[key]
    return None

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

images, csvs = list_dataset_files(RAW_DATA_DIR)
image_index = build_image_index(images)
df_raw = pd.read_csv(METADATA_CSV)

records = []
for i, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Ánh xạ metadata"):
    img_path = resolve_image_path(row[IMAGE_COLUMN], image_index)
    label = normalize_label(row[LABEL_COLUMN])
    image_id = str(row[IMAGE_COLUMN]).strip()
    source = str(row[SOURCE_COLUMN]).strip() if SOURCE_COLUMN else "unknown"
    patient_id = str(row[PATIENT_COLUMN]).strip() if PATIENT_COLUMN else ""
    eye = str(row[EYE_COLUMN]).strip() if EYE_COLUMN else ""
    existing_split = str(row[SPLIT_COLUMN]).strip().lower() if SPLIT_COLUMN else ""
    records.append({
        "image_id": image_id,
        "original_path": str(img_path) if img_path else "",
        "label": label,
        "source": source if source and source != "nan" else "unknown",
        "patient_id": patient_id if patient_id and patient_id != "nan" else "",
        "eye": eye if eye and eye != "nan" else "",
        "existing_split": existing_split if existing_split in {"train", "val", "test"} else "",
        "metadata_row": i,
    })

df = pd.DataFrame(records)
df["has_file"] = df["original_path"].astype(bool)
df["has_valid_label"] = df["label"].isin([0, 1, 2, 3, 4])

unmatched_metadata = df[~df["has_file"]]
invalid_labels = df[~df["has_valid_label"]]
metadata_paths = set(df.loc[df["has_file"], "original_path"])
unlabeled_images = [str(p) for p in images if str(p) not in metadata_paths]

REPORT_DIR.mkdir(parents=True, exist_ok=True)
unmatched_metadata.to_csv(REPORT_DIR / "metadata_without_image.csv", index=False)
invalid_labels.to_csv(REPORT_DIR / "invalid_labels.csv", index=False)
pd.DataFrame({"image_path": unlabeled_images}).to_csv(REPORT_DIR / "images_without_metadata.csv", index=False)

df = df[df["has_file"] & df["has_valid_label"]].copy()
df["label"] = df["label"].astype(int)

print("Phân bố nhãn hợp lệ:")
display(df["label"].value_counts().sort_index())
print("Ảnh có metadata hợp lệ:", len(df))
if not PATIENT_COLUMN:
    print("CẢNH BÁO: Không có patient_id, split không thể chống leakage theo bệnh nhân.")

# ## 6. Kiểm tra ảnh lỗi

def read_rgb(path: Path) -> np.ndarray:
    """Đọc ảnh bằng PIL và chuyển an toàn về RGB uint8."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im)
    return arr.astype(np.uint8)

def black_pixel_ratio_rgb(img: np.ndarray, threshold: int = BLACK_THRESHOLD) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float((gray <= threshold).mean())

def validate_image(path: Path) -> Dict[str, Any]:
    result = {
        "original_path": str(path),
        "error_type": "",
        "error_message": "",
        "width": None,
        "height": None,
        "channels": None,
        "black_pixel_ratio": None,
    }
    try:
        with Image.open(path) as im:
            result["width"], result["height"] = im.size
            bands = im.getbands()
            result["channels"] = len(bands)
            if im.size[0] <= 0 or im.size[1] <= 0:
                raise ValueError("width_or_height_zero")
            if min(im.size) < MIN_IMAGE_SIDE:
                raise ValueError("too_small")
            if "A" in bands:
                result["error_type"] = "has_alpha_channel"
            img = np.asarray(im.convert("RGB"))
        result["black_pixel_ratio"] = black_pixel_ratio_rgb(img)
        if np.all(img == img.reshape(-1, 3)[0]):
            raise ValueError("single_color")
        if result["black_pixel_ratio"] > 0.98:
            raise ValueError("too_much_black")
    except Exception as e:
        if not result["error_type"]:
            result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
    return result

validation_rows = []
for p in tqdm(df["original_path"].map(Path), desc="Kiểm tra ảnh lỗi"):
    validation_rows.append(validate_image(p))

invalid_df = pd.DataFrame(validation_rows)
invalid_df = invalid_df[invalid_df["error_type"].astype(bool)].copy()
invalid_df.to_csv(REPORT_DIR / "invalid_images.csv", index=False)
print("Số ảnh cảnh báo/lỗi:", len(invalid_df))
display(invalid_df.head())

valid_paths = set(df["original_path"]) - set(invalid_df.loc[invalid_df["error_message"].astype(bool), "original_path"])
df = df[df["original_path"].isin(valid_paths)].copy()

# ## 7. Phát hiện ảnh trùng exact hash và gần trùng bằng pHash

def compute_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path_str in tqdm(frame["original_path"], desc="Tính hash"):
        p = Path(path_str)
        row = {"original_path": path_str, "exact_hash": "", "perceptual_hash": ""}
        try:
            row["exact_hash"] = sha256_file(p)
            if ENABLE_PERCEPTUAL_DUPLICATE_CHECK:
                with Image.open(p) as im:
                    row["perceptual_hash"] = str(imagehash.phash(im.convert("RGB")))
        except Exception as e:
            row["hash_error"] = str(e)
        rows.append(row)
    return pd.DataFrame(rows)

hash_df = compute_hashes(df)
df = df.merge(hash_df, on="original_path", how="left")

dup_rows = []
for h, g in df.groupby("exact_hash"):
    if h and len(g) > 1:
        paths = list(g["original_path"])
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                dup_rows.append({
                    "image_path_1": paths[i],
                    "image_path_2": paths[j],
                    "exact_hash": h,
                    "perceptual_hash": "",
                    "hash_distance": 0,
                    "duplicate_type": "exact",
                })

if ENABLE_PERCEPTUAL_DUPLICATE_CHECK:
    ph = df[["original_path", "perceptual_hash"]].dropna()
    ph = ph[ph["perceptual_hash"].astype(bool)]
    phashes = [(r.original_path, imagehash.hex_to_hash(r.perceptual_hash)) for r in ph.itertuples()]
    for i in tqdm(range(len(phashes)), desc="So pHash gần trùng"):
        p1, h1 = phashes[i]
        for j in range(i + 1, len(phashes)):
            p2, h2 = phashes[j]
            dist = h1 - h2
            if 0 < dist <= PHASH_DISTANCE_THRESHOLD:
                dup_rows.append({
                    "image_path_1": p1,
                    "image_path_2": p2,
                    "exact_hash": "",
                    "perceptual_hash": str(h1),
                    "hash_distance": dist,
                    "duplicate_type": "near",
                })

duplicate_df = pd.DataFrame(dup_rows)
duplicate_df.to_csv(REPORT_DIR / "duplicate_images.csv", index=False)
print("Số cặp trùng/gần trùng:", len(duplicate_df))
display(duplicate_df.head())

# ## 8. Chia train/validation/test trước khi tiền xử lý
#
# Nếu có `patient_id`, notebook ưu tiên group split để ảnh cùng bệnh nhân không rơi vào nhiều split. Nếu không có, dùng stratified split theo label và ghi cảnh báo.

def stratified_split_no_group(frame: pd.DataFrame) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        frame,
        train_size=TRAIN_RATIO,
        stratify=frame["label"],
        random_state=RANDOM_STATE,
    )
    rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)

def group_split(frame: pd.DataFrame) -> pd.DataFrame:
    groups = frame["patient_id"].fillna("").astype(str)
    if (groups == "").all():
        return stratified_split_no_group(frame)
    # StratifiedGroupKFold gần đúng tỷ lệ 70/15/15, ưu tiên chống leakage bệnh nhân.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = frame["label"].values
    idx = np.arange(len(frame))
    folds = list(sgkf.split(idx, y, groups))
    test_idx = folds[0][1]
    remaining = frame.drop(frame.index[test_idx]).reset_index(drop=True)
    test = frame.iloc[test_idx].copy()
    groups2 = remaining["patient_id"].fillna("").astype(str)
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    folds2 = list(sgkf2.split(np.arange(len(remaining)), remaining["label"].values, groups2))
    val_idx = folds2[0][1]
    val = remaining.iloc[val_idx].copy()
    train = remaining.drop(remaining.index[val_idx]).copy()
    train["split"] = "train"; val["split"] = "val"; test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)

if USE_EXISTING_SPLIT_IF_AVAILABLE and "existing_split" in df.columns and df["existing_split"].isin(["train", "val", "test"]).any():
    split_df = df[df["existing_split"].isin(["train", "val", "test"])].copy()
    split_df["split"] = split_df["existing_split"]
    logger.info("Dùng split có sẵn từ folder dataset, không chia lại train/val/test.")
else:
    split_df = group_split(df)
split_cols = ["image_id", "original_path", "label", "split", "patient_id", "source", "exact_hash"]
split_df[split_cols].to_csv(DRIVE_OUTPUT_DIR / "data_split.csv", index=False)

print("Phân bố class theo split:")
display(pd.crosstab(split_df["split"], split_df["label"]))
print("Phân bố source theo split:")
display(pd.crosstab(split_df["split"], split_df["source"]))

# Cảnh báo exact hash rơi qua nhiều split.
leak = split_df.groupby("exact_hash")["split"].nunique()
leak_hashes = leak[leak > 1].index.tolist()
if leak_hashes:
    print("CẢNH BÁO leakage exact hash giữa các split:", len(leak_hashes))

# ## 9. Hàm crop fundus, resize padding, Ben Graham và CLAHE

def crop_fundus(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop vùng fundus sáng lớn nhất, thêm margin và fallback về ảnh gốc nếu thất bại."""
    img = image.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    info = {
        "crop_x1": 0, "crop_y1": 0, "crop_x2": w, "crop_y2": h,
        "cropped_width": w, "cropped_height": h,
        "crop_status": "fallback",
        "black_pixel_ratio": black_pixel_ratio_rgb(img),
    }
    if not contours:
        return img, info

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area / float(h * w) < MIN_FUNDUS_AREA_RATIO:
        return img, info

    x, y, bw, bh = cv2.boundingRect(c)
    margin = int(max(bw, bh) * CROP_MARGIN_RATIO)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    cropped = img[y1:y2, x1:x2]
    info.update({
        "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
        "cropped_width": int(x2 - x1), "cropped_height": int(y2 - y1),
        "crop_status": "ok",
        "black_pixel_ratio": black_pixel_ratio_rgb(cropped),
    })
    return cropped, info

def resize_with_padding(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Resize giữ tỷ lệ và padding đen về target_size x target_size RGB uint8."""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x = (target_size - nw) // 2
    y = (target_size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas

def ben_graham_preprocess(image: np.ndarray) -> np.ndarray:
    """Ben Graham preprocessing giữ RGB, sigma theo kích thước ảnh."""
    img = image.astype(np.float32)
    sigma = max(image.shape[:2]) * BEN_GRAHAM_SIGMA_RATIO
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    processed = cv2.addWeighted(img, BEN_GRAHAM_ALPHA, blurred, BEN_GRAHAM_BETA, BEN_GRAHAM_GAMMA)
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = (gray > BLACK_THRESHOLD).astype(np.uint8)
    processed[mask == 0] = 0
    return processed

def clahe_lab_rgb(image: np.ndarray) -> np.ndarray:
    """CLAHE tùy chọn trên kênh L trong LAB, không áp dụng độc lập trên RGB."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

def save_rgb_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(image.astype(np.uint8), mode="RGB")
    if OUTPUT_FORMAT.lower() in ["jpg", "jpeg"]:
        im.save(path, quality=JPEG_QUALITY, subsampling=0, optimize=True)
    else:
        im.save(path)

def safe_name(row: pd.Series) -> str:
    source = str(row.get("source", "unknown")).strip().lower().replace(" ", "_") or "unknown"
    image_id = Path(str(row["image_id"])).stem.replace(" ", "_")
    short_hash = str(row.get("exact_hash", ""))[:8] or hashlib.sha1(str(row["original_path"]).encode()).hexdigest()[:8]
    return f"{source}_{image_id}_{short_hash}.{OUTPUT_FORMAT}"

def chunk_name(chunk_id: int) -> str:
    return f"chunk_{chunk_id:04d}"

# ## 10. Pipeline xử lý toàn bộ dataset có checkpoint/resume
#
# Khi `DRY_RUN=True`, notebook chỉ xử lý khoảng `DRY_RUN_IMAGES` ảnh để bạn kiểm tra bằng mắt. Sau khi ổn, đặt `DRY_RUN=False` và chạy lại.

def process_one(row: pd.Series) -> Dict[str, Any]:
    out_name = safe_name(row)
    label = str(int(row["label"]))
    split = row["split"]
    chunk_dir = chunk_name(int(row.get("chunk_id", 0)))
    base_dir = PROCESSED_DATA_DIR / "rgb_crop_512" / chunk_dir / split / label
    bg_dir = PROCESSED_DATA_DIR / "rgb_bengraham_512" / chunk_dir / split / label
    clahe_dir = PROCESSED_DATA_DIR / "rgb_clahe_512" / chunk_dir / split / label
    base_path = base_dir / out_name
    bg_path = bg_dir / out_name
    clahe_path = clahe_dir / out_name

    manifest = {
        "image_id": row["image_id"],
        "original_path": row["original_path"],
        "baseline_processed_path": str(base_path),
        "bengraham_processed_path": str(bg_path),
        "clahe_processed_path": str(clahe_path) if ENABLE_CLAHE_DATASET else "",
        "label": row["label"],
        "split": row["split"],
        "chunk_id": int(row.get("chunk_id", 0)),
        "source": row.get("source", "unknown"),
        "patient_id": row.get("patient_id", ""),
        "original_width": None,
        "original_height": None,
        "cropped_width": None,
        "cropped_height": None,
        "output_width": TARGET_SIZE,
        "output_height": TARGET_SIZE,
        "crop_x1": None, "crop_y1": None, "crop_x2": None, "crop_y2": None,
        "black_pixel_ratio": None,
        "exact_hash": row.get("exact_hash", ""),
        "processing_status": "pending",
        "error_message": "",
    }

    if RESUME_PROCESSING and not OVERWRITE_EXISTING and base_path.exists() and bg_path.exists():
        manifest["processing_status"] = "skipped_existing"
        return manifest

    try:
        img = read_rgb(Path(row["original_path"]))
        manifest["original_height"], manifest["original_width"] = img.shape[:2]
        cropped, crop_info = crop_fundus(img)
        manifest.update(crop_info)
        baseline = resize_with_padding(cropped, TARGET_SIZE)
        bg_input = cropped
        bg = ben_graham_preprocess(bg_input)
        bg = resize_with_padding(bg, TARGET_SIZE)

        if baseline.shape != (TARGET_SIZE, TARGET_SIZE, 3) or bg.shape != (TARGET_SIZE, TARGET_SIZE, 3):
            raise ValueError("Output shape không đúng 512x512x3")

        save_rgb_image(baseline, base_path)
        save_rgb_image(bg, bg_path)
        if ENABLE_CLAHE_DATASET:
            clahe = resize_with_padding(clahe_lab_rgb(cropped), TARGET_SIZE)
            save_rgb_image(clahe, clahe_path)

        manifest["processing_status"] = "success"
    except Exception as e:
        manifest["processing_status"] = "error"
        manifest["error_message"] = str(e)
    return manifest

def prepare_work_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn bị danh sách ảnh cần xử lý và gán chunk_id."""
    work = frame.copy().reset_index(drop=True)
    if DRY_RUN:
        work = work.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, DRY_RUN_IMAGES // 5)), random_state=RANDOM_STATE)
        ).head(DRY_RUN_IMAGES).reset_index(drop=True)
        logger.info("DRY_RUN=True, chỉ xử lý %s ảnh.", len(work))
    work["chunk_id"] = (np.arange(len(work)) // CHUNK_SIZE) + 1
    return work

def zip_chunk_outputs(chunk_id: int) -> Optional[Path]:
    """Nén output chunk trên /content rồi copy zip sang Google Drive."""
    import zipfile

    name = chunk_name(chunk_id)
    local_zip_dir = PROCESSED_DATA_DIR / "chunks_zipped"
    drive_zip_dir = DRIVE_OUTPUT_DIR / "chunks_zipped"
    local_zip_dir.mkdir(parents=True, exist_ok=True)
    drive_zip_dir.mkdir(parents=True, exist_ok=True)
    local_zip_path = local_zip_dir / f"{name}.zip"
    drive_zip_path = drive_zip_dir / f"{name}.zip"

    paths_to_zip = [
        PROCESSED_DATA_DIR / "rgb_crop_512" / name,
        PROCESSED_DATA_DIR / "rgb_bengraham_512" / name,
    ]
    if ENABLE_CLAHE_DATASET:
        paths_to_zip.append(PROCESSED_DATA_DIR / "rgb_clahe_512" / name)

    files = []
    for src in paths_to_zip:
        if src.exists():
            files.extend([p for p in src.rglob("*") if p.is_file()])

    chunk_manifest = DRIVE_OUTPUT_DIR / f"processed_manifest_{name}.csv"
    if chunk_manifest.exists():
        files.append(chunk_manifest)

    if not files:
        logger.warning("Không có dữ liệu để nén cho %s.", name)
        return None

    with zipfile.ZipFile(local_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in tqdm(files, desc=f"Nén {name}"):
            if str(p).startswith(str(PROCESSED_DATA_DIR)):
                arcname = p.relative_to(PROCESSED_DATA_DIR)
            else:
                arcname = Path(p.name)
            zf.write(p, arcname=arcname)

    shutil.copy2(local_zip_path, drive_zip_path)
    logger.info("Đã nén %s local và copy sang Drive: %s", name, drive_zip_path)

    if DELETE_UNCOMPRESSED_CHUNK_AFTER_ZIP:
        for src in paths_to_zip:
            if src.exists():
                shutil.rmtree(src)
        logger.info("Đã xóa thư mục ảnh chưa nén của %s để tiết kiệm dung lượng /content.", name)
    if local_zip_path.exists():
        local_zip_path.unlink()

    return drive_zip_path

def validate_chunk_outputs(manifest: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
    """QC output while local chunk files still exist, before they are zipped/deleted."""
    quality_rows = []
    successful = manifest[manifest["processing_status"].isin(["success", "skipped_existing"])]
    for column in ["baseline_processed_path", "bengraham_processed_path"]:
        if column not in successful.columns:
            continue
        for path_value in tqdm(successful[column].dropna(), desc=f"QC {chunk_name(chunk_id)} {column}"):
            path = Path(path_value)
            result = {"path": str(path), "ok": False, "width": None, "height": None,
                      "channels": None, "error": "", "version": column,
                      "chunk_id": chunk_id}
            try:
                with Image.open(path) as image:
                    result["width"], result["height"] = image.size
                    result["channels"] = len(image.convert("RGB").getbands())
                    result["ok"] = (
                        image.size == (TARGET_SIZE, TARGET_SIZE)
                        and result["channels"] == 3
                    )
                    if not result["ok"]:
                        result["error"] = "invalid_size_or_channels"
            except Exception as exc:
                result["error"] = str(exc)
            quality_rows.append(result)

    quality = pd.DataFrame(quality_rows)
    quality_path = REPORT_DIR / f"output_quality_{chunk_name(chunk_id)}.csv"
    quality.to_csv(quality_path, index=False)
    failed = int((~quality["ok"]).sum()) if not quality.empty else 0
    if failed:
        raise RuntimeError(
            f"{chunk_name(chunk_id)} có {failed} output lỗi; giữ file local để kiểm tra, chưa nén/xóa."
        )
    logger.info("QC %s thành công: %s output.", chunk_name(chunk_id), len(quality))
    return quality

def run_processing_chunk(chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
    """Xử lý một chunk và ghi manifest riêng cho chunk đó."""
    name = chunk_name(chunk_id)
    chunk_manifest_path = DRIVE_OUTPUT_DIR / f"processed_manifest_{name}.csv"
    archived_chunk = DRIVE_OUTPUT_DIR / "chunks_zipped" / f"{name}.zip"
    rows = []

    if RESUME_PROCESSING and chunk_manifest_path.exists():
        old_manifest = pd.read_csv(chunk_manifest_path)
        ok = old_manifest[old_manifest["processing_status"].isin(["success", "skipped_existing"])].copy()
        done_keys = set(ok["original_path"].astype(str))
        expected_keys = set(chunk["original_path"].astype(str))
        completed_current_chunk = expected_keys.issubset(done_keys)

        if completed_current_chunk and archived_chunk.exists():
            logger.info("%s đã được QC và lưu tại %s; bỏ qua xử lý lại.", name, archived_chunk)
            return old_manifest

        if archived_chunk.exists():
            logger.warning(
                "%s có zip cũ nhưng manifest không khớp chunk hiện tại "
                "(có thể là bản DRY_RUN). Sẽ xử lý lại toàn bộ %s ảnh và ghi đè zip.",
                name, len(chunk)
            )
            rows = []
        else:
            rows.extend(old_manifest.to_dict("records"))
            chunk = chunk[~chunk["original_path"].astype(str).isin(done_keys)].reset_index(drop=True)
            logger.info("%s resume: đã có %s ảnh thành công, còn %s ảnh.", name, len(done_keys), len(chunk))

    for i, (_, row) in enumerate(tqdm(chunk.iterrows(), total=len(chunk), desc=f"Tiền xử lý {name}")):
        rows.append(process_one(row))
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            pd.DataFrame(rows).to_csv(chunk_manifest_path, index=False)
            logger.info("%s checkpoint tại %s ảnh.", name, i + 1)
            print_storage_report()

    out = pd.DataFrame(rows)
    out.to_csv(chunk_manifest_path, index=False)
    validate_chunk_outputs(out, chunk_id)
    if ZIP_EACH_CHUNK:
        zip_chunk_outputs(chunk_id)
    return out

def run_processing(frame: pd.DataFrame) -> pd.DataFrame:
    """Xử lý dataset theo chunk 10k ảnh để phù hợp Colab Free."""
    import gc

    work = prepare_work_frame(frame)
    manifest_path = DRIVE_OUTPUT_DIR / "processed_manifest.csv"
    all_rows = []

    if not PROCESS_IN_CHUNKS:
        work["chunk_id"] = 1

    for chunk_id in sorted(work["chunk_id"].unique()):
        chunk = work[work["chunk_id"] == chunk_id].reset_index(drop=True)
        logger.info("Bắt đầu %s với %s ảnh.", chunk_name(int(chunk_id)), len(chunk))
        chunk_result = run_processing_chunk(chunk, int(chunk_id))
        all_rows.append(chunk_result)

        combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        combined.to_csv(manifest_path, index=False)

        del chunk
        del chunk_result
        gc.collect()
        print_storage_report()

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(manifest_path, index=False)
    return out

manifest_df = run_processing(split_df)
display(manifest_df["processing_status"].value_counts())
display(manifest_df.head())

# ## 11. Kiểm tra chất lượng dataset đầu ra và tạo manual_review.csv

def check_output_image(path: str) -> Dict[str, Any]:
    r = {"path": path, "ok": False, "width": None, "height": None, "channels": None, "error": ""}
    try:
        with Image.open(path) as im:
            r["width"], r["height"] = im.size
            r["channels"] = len(im.getbands())
            r["ok"] = (im.size == (TARGET_SIZE, TARGET_SIZE) and len(im.convert("RGB").getbands()) == 3)
    except Exception as e:
        r["error"] = str(e)
    return r

success_manifest = manifest_df[manifest_df["processing_status"].isin(["success", "skipped_existing"])].copy()
chunk_quality_files = sorted(REPORT_DIR.glob("output_quality_chunk_*.csv"))
quality_df = pd.concat(
    [pd.read_csv(path) for path in chunk_quality_files], ignore_index=True
) if chunk_quality_files else pd.DataFrame()
quality_df.to_csv(REPORT_DIR / "output_quality_report.csv", index=False)
print("QC lỗi:")
display(quality_df[~quality_df["ok"]].head() if not quality_df.empty else quality_df)

manual = success_manifest[
    (success_manifest["processing_status"] != "success") |
    (success_manifest["black_pixel_ratio"].fillna(0) > MAX_BLACK_RATIO_WARN) |
    ((success_manifest["cropped_width"].fillna(TARGET_SIZE) * success_manifest["cropped_height"].fillna(TARGET_SIZE)) <
     (TARGET_SIZE * TARGET_SIZE * 0.10))
].copy()
manual.to_csv(REPORT_DIR / "manual_review.csv", index=False)
print("Số ảnh cần review thủ công:", len(manual))

print("Số lượng theo split/class:")
display(pd.crosstab(success_manifest["split"], success_manifest["label"]))
print("Tỷ lệ xử lý thành công:", (manifest_df["processing_status"] == "success").mean())

# ## 12. Trực quan hóa trước/sau và histogram RGB

def show_samples(manifest: pd.DataFrame, n: int = 10) -> None:
    sample = manifest[manifest["processing_status"].isin(["success", "skipped_existing"])].copy()
    sample = sample[
        sample["baseline_processed_path"].map(lambda p: Path(p).exists())
        & sample["bengraham_processed_path"].map(lambda p: Path(p).exists())
    ]
    if sample.empty:
        print("Ảnh local đã được QC, nén sang Drive và xóa để tiết kiệm dung lượng; bỏ qua hiển thị mẫu.")
        return
    sample = sample.sample(min(n, len(sample)), random_state=RANDOM_STATE)
    for _, r in sample.iterrows():
        orig = read_rgb(Path(r["original_path"]))
        base = read_rgb(Path(r["baseline_processed_path"]))
        bg = read_rgb(Path(r["bengraham_processed_path"]))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, img, title in zip(axes, [orig, base, bg], ["Gốc", "RGB crop", "Ben Graham"]):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(
            f"label={r['label']} | source={r['source']} | split={r['split']} | "
            f"orig={r['original_width']}x{r['original_height']} | "
            f"bbox=({r['crop_x1']},{r['crop_y1']},{r['crop_x2']},{r['crop_y2']}) | "
            f"black={r['black_pixel_ratio']:.3f}"
        )
        plt.show()

        plt.figure(figsize=(10, 3))
        for img, name in [(orig, "Gốc"), (base, "Crop"), (bg, "Ben Graham")]:
            for c, color in enumerate(["r", "g", "b"]):
                hist = cv2.calcHist([img], [c], None, [256], [0, 256]).flatten()
                plt.plot(hist, color=color, alpha=0.25, label=f"{name}-{color}")
        plt.title("Histogram RGB")
        plt.show()

show_samples(manifest_df, n=10)

# ## 13. Kiểm tra tương thích EfficientNetB3
#
# Dataset offline vẫn lưu 512x512. Khi training, resize về kích thước model trong pipeline `image_dataset_from_directory`. EfficientNet của `tf.keras.applications` đã có preprocessing tích hợp trong model gốc; tránh rescale hai lần nếu bạn dùng đúng API.

import tensorflow as tf

def test_efficientnet_input(version_dir: Path, image_size: Tuple[int, int] = (300, 300)) -> None:
    train_dir = version_dir / "train"
    if not train_dir.exists():
        chunk_train_dirs = sorted(version_dir.glob("chunk_*/train"))
        train_dir = chunk_train_dirs[0] if chunk_train_dirs else train_dir
    if not train_dir.exists():
        print("Không thấy train_dir:", train_dir)
        print("Do DELETE_UNCOMPRESSED_CHUNK_AFTER_ZIP=True, output local có thể đã bị xóa sau khi nén.")
        print("Hãy giải nén một file trong DRIVE_OUTPUT_DIR/chunks_zipped để test EfficientNet.")
        return
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=image_size,
        batch_size=8,
        shuffle=True,
        seed=RANDOM_STATE,
    )
    x, y = next(iter(ds))
    print("Batch shape:", x.shape)
    print("Batch dtype:", x.dtype)
    print("Min/max:", float(tf.reduce_min(x)), float(tf.reduce_max(x)))
    print("Labels:", y.numpy())
    model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        pooling="avg",
    )
    out = model(x, training=False)
    print("EfficientNetB3 output shape:", out.shape)

test_efficientnet_input(PROCESSED_DATA_DIR / "rgb_bengraham_512")

# ## 14. Nén hoặc lưu dataset trên Google Drive
#
# Toàn bộ output đã nằm trong `PROCESSED_DATA_DIR`. Nếu muốn tạo file zip để tải về/chia sẻ, bật cell dưới.

CREATE_ZIP = False
if CREATE_ZIP:
    zip_base = PROJECT_DIR / "processed_dataset"
    shutil.make_archive(str(zip_base), "zip", root_dir=PROCESSED_DATA_DIR)
    print("Đã tạo:", str(zip_base) + ".zip")


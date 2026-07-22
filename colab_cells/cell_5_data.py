# ════════════════════════════════════════════════════════════════
# Cell 5: Tải dataset Kaggle + tạo manifest + cấu hình huấn luyện
# ════════════════════════════════════════════════════════════════
import getpass, shutil, zipfile
import pandas as pd
from google.colab import userdata

IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}

def get_kaggle_token():
    try:
        token = userdata.get('KAGGLE_API_TOKEN')
    except Exception:
        token = getpass.getpass('Dán Kaggle API token: ').strip()
    if not token:
        raise RuntimeError('Chưa cung cấp KAGGLE_API_TOKEN')
    os.environ['KAGGLE_API_TOKEN'] = token

def download_kaggle_source(dataset_ref, extract_name):
    target = Path('/content/selected_data') / extract_name
    marker = target / '.kaggle_source'
    if marker.is_file() and marker.read_text(encoding='utf-8').strip() == dataset_ref:
        print(f'Dùng lại dataset Kaggle đã tải: {target}')
        return target.resolve()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    get_kaggle_token()
    kaggle_cli = [sys.executable, '-m', 'kaggle']
    subprocess.run(
        kaggle_cli + ['datasets', 'files', '-d', dataset_ref, '--page-size', '20'],
        check=True, capture_output=True, text=True,
    )
    print(f'Đang tải Kaggle dataset {dataset_ref}...')
    subprocess.run(kaggle_cli + ['datasets', 'download', '-d', dataset_ref, '-p', str(target)], check=True)
    archives = sorted(target.glob('*.zip'))
    if not archives:
        raise FileNotFoundError(f'Kaggle không tạo ZIP trong {target}')
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target)
        archive_path.unlink()
    marker.write_text(dataset_ref, encoding='utf-8')
    return target.resolve()

def find_labeled_train_dir(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    candidates = [dataset_root]
    candidates.extend(
        path for path in dataset_root.rglob('*')
        if path.is_dir() and path.name.lower() in {'train', 'training'}
    )
    for candidate in candidates:
        class_dirs = {
            path.name: path for path in candidate.iterdir()
            if path.is_dir() and path.name in {'0', '1', '2', '3', '4'}
        }
        if set(class_dirs) == {'0', '1', '2', '3', '4'}:
            return candidate, class_dirs
    raise ValueError(f'Không tìm thấy train/0..4 hoặc training/0..4 bên dưới {dataset_root}')

def create_labeled_manifest(dataset_root, csv_path):
    train_dir, class_dirs = find_labeled_train_dir(dataset_root)
    records = []
    for label in range(5):
        for image_path in sorted(class_dirs[str(label)].rglob('*')):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({'image_path': str(image_path.resolve()), 'diagnosis': label})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f'Không tìm thấy ảnh có nhãn trong {train_dir}')
    missing = sorted(set(range(5)) - set(frame['diagnosis'].unique()))
    if missing:
        raise ValueError(f'Train thiếu ảnh cho các lớp: {missing}')
    frame.to_csv(csv_path, index=False)
    return train_dir, frame

# ── Thực thi ──
CHECKPOINT_PATH = Path(checkpoint_widget.value).expanduser().resolve()
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f'Checkpoint không tồn tại: {CHECKPOINT_PATH}')

LABELED_ROOT = download_kaggle_source(LABELED_KAGGLE_DATASET, 'labeled_dataset')
UNLABELED_DIR = download_kaggle_source(UNLABELED_KAGGLE_DATASET, 'unlabeled_dataset')

OUTPUT_DIR = ARTIFACT_PATH.parent
LABELED_CSV = OUTPUT_DIR / 'labeled_train.csv'
TRAIN_DIR, labeled_df = create_labeled_manifest(LABELED_ROOT, LABELED_CSV)
unlabeled_count = sum(
    1 for path in UNLABELED_DIR.rglob('*')
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)
if unlabeled_count == 0:
    raise ValueError(f'Không tìm thấy ảnh chưa nhãn trong {UNLABELED_DIR}')

SEMI_CONFIG = {
    'threshold': 0.95,
    'pseudo_weight': 0.25,
    'epochs': 10,
    'batch_size': 16,
    'lr': 1e-4,
}
print(f'Checkpoint     : {CHECKPOINT_PATH}')
print(f'Labeled train  : {TRAIN_DIR} ({len(labeled_df):,} ảnh)')
print(f'Unlabeled      : {UNLABELED_DIR} ({unlabeled_count:,} ảnh)')
print(f'Output model   : {ARTIFACT_PATH}')
print('Cấu hình       :', SEMI_CONFIG)
display(labeled_df.groupby('diagnosis').size().rename('images').to_frame())

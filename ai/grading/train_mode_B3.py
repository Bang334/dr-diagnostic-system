import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import glob
import pickle
import hashlib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
# Enable GPU memory growth to avoid OOM and hanging on Colab free GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] Could not set memory growth: {e}")

# ============================================================
#  SEED & XLA — Reproducibility + Tối ưu graph
# ============================================================
tf.keras.utils.set_random_seed(42)
print("[INFO] Random seed set to 42 (numpy, tf, python random)")

tf.config.optimizer.set_jit(True)
print("[INFO] XLA JIT compilation enabled")

from tensorflow.keras.applications import EfficientNetB3, ResNet50, ConvNeXtTiny
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)

# ============================================================
#  BƯỚC 0: MIXED PRECISION (Train nhanh hơn, tăng batch size)
# ============================================================
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("[INFO] Mixed Precision: float16 enabled")

# ============================================================
#  BƯỚC 1: MOUNT GOOGLE DRIVE & CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
try:
    import google.colab
    if not os.path.exists('/content/drive/MyDrive'):
        print("[INFO] Đang chạy trên Colab, thử mount Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
    else:
        print("[INFO] Google Drive đã được kết nối sẵn.")
except ImportError:
    print("[INFO] Không chạy trên Google Colab, bỏ qua bước mount Drive.")

DRIVE_DIR = "/content/drive/MyDrive/train model"
DATASET_ZIP = "/content/drive/MyDrive/dataset_split_rgb.zip"

EXTRACT_DIR = "/content/dataset_split_rgb"
TRAIN_DIR = f"{EXTRACT_DIR}/train"
VAL_DIR = f"{EXTRACT_DIR}/val"
TEST_DIR = f"{EXTRACT_DIR}/test"

MODELS_DIR = f"{DRIVE_DIR}/models"

# Cập nhật đường dẫn cục bộ nếu chạy trên PC
if not os.path.exists('/content/'):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, 'dataset_bengraham', 'train')
    VAL_DIR = os.path.join(BASE_DIR, 'dataset_bengraham', 'val')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

if os.path.exists('/content/'):
    if not os.path.exists(EXTRACT_DIR):
        if not os.path.exists(DATASET_ZIP):
            raise FileNotFoundError(f"[LỖI CRITICAL] Không tìm thấy file dataset tại '{DATASET_ZIP}'.")
        import zipfile
        print("[DATA] Đang giải nén dataset...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
            zf.extractall("/content/")

# ============================================================
#  CẤU HÌNH TRAINING
# ============================================================
# Lựa chọn mô hình: 'EfficientNetB3', 'ResNet50', hoặc 'ConvNeXtTiny'
MODEL_NAME = "EfficientNetB3"
EXPERIMENT_NAME = f"{MODEL_NAME}_rgb_crop_v1"

# Tự động cấu hình kích thước ảnh tùy theo mô hình
if MODEL_NAME == 'EfficientNetB3':
    IMG_SIZE = 384
elif MODEL_NAME == 'ResNet50':
    IMG_SIZE = 224
elif MODEL_NAME == 'ConvNeXtTiny':
    IMG_SIZE = 224
else:
    IMG_SIZE = 224 # Default

BATCH_SIZE  = 16   # Default batch size (adjustable 32–64 depending on GPU)
NUM_CLASSES = 5
EPOCHS_HEAD = 8    # Warm‑up epochs for head
EPOCHS_TUNE = 25   # Giảm xuống 25; EarlyStopping sẽ dừng sớm nếu hội tụ — tránh timeout 12h
LABEL_SMOOTHING = 0.05
TTA_AUGMENTATIONS = 2  # Validation và test phải dùng cùng một quy trình suy luận
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.70
THRESHOLD_STEPS = 81
USE_CLASS_WEIGHTS = False
EVALUATE_ONLY = True  # Load best checkpoint and tune/evaluate thresholds without training.
CLASS_WEIGHT_POWER = 0.5  # sqrt inverse frequency is more stable than full inverse frequency
MAX_CLASS_WEIGHT = 5.0
AUDIT_EXACT_DUPLICATES = False  # True is thorough but reads every image once

# ============================================================
#  BƯỚC 2: CORAL ORDINAL LOSS & UTILITIES
# ============================================================
def ordinal_loss(y_true, y_pred):
    """
    CORAL-style Ordinal Regression Loss.
    Chuyển label integer (0-4) thành cumulative labels rồi tính binary cross-entropy.
    Ví dụ: label=2 → [1, 1, 0, 0] (P(Y>0)=1, P(Y>1)=1, P(Y>2)=0, P(Y>3)=0)
    """
    y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
    # Tạo cumulative labels: label=k → [1]*k + [0]*(NUM_CLASSES-1-k)
    thresholds = tf.cast(tf.range(1, NUM_CLASSES), tf.float32)  # [1, 2, 3, 4]
    cumulative = tf.cast(y_true >= thresholds, tf.float32)
    # Optional label smoothing
    cumulative = cumulative * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
    # Binary cross-entropy per threshold
    loss = tf.keras.losses.binary_crossentropy(cumulative, y_pred)
    # Keep one loss value per image so Keras can apply sample_weight correctly.
    return loss

def ordinal_to_class(ordinal_probs):
    """
    Convert ordinal probabilities → class label.
    P(Y>k) > 0.5 thì count thêm 1 bậc.
    Ví dụ: [0.95, 0.8, 0.3, 0.05] → class 2
    """
    return tf.reduce_sum(tf.cast(ordinal_probs > 0.5, tf.int32), axis=-1)

# ============================================================
#  BƯỚC 3: CHUẨN BỊ DỮ LIỆU TỪ THƯ MỤC TRAIN VÀ VAL
# ============================================================
print("\n[DATA] Đang nạp danh sách file ảnh từ thư mục...")
def get_paths_and_labels(data_dir):
    paths = []
    labels = []
    if os.path.exists(data_dir):
        for label_dir in os.listdir(data_dir):
            full_label_dir = os.path.join(data_dir, label_dir)
            if os.path.isdir(full_label_dir):
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
                    found = glob.glob(os.path.join(full_label_dir, ext))
                    paths.extend(found)
                    labels.extend([int(label_dir)] * len(found))
    # Chuyển về mảng với kiểu dữ liệu rõ ràng (tránh lỗi float64 khi mảng rỗng)
    return np.array(paths, dtype=str), np.array(labels, dtype=np.int32)

train_paths, train_labels = get_paths_and_labels(TRAIN_DIR)
val_paths, val_labels   = get_paths_and_labels(VAL_DIR)
test_paths, test_labels = get_paths_and_labels(TEST_DIR)

print(f"[DATA] Tập Train: {len(train_paths)} ảnh.")
print(f"[DATA] Tập Val:   {len(val_paths)} ảnh.")
print(f"[DATA] Tập Test:  {len(test_paths)} ảnh.")

def print_class_distribution(name, labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = max(int(counts.sum()), 1)
    details = ", ".join(
        f"class {i}: {count} ({count / total:.1%})"
        for i, count in enumerate(counts)
    )
    print(f"[DATA] {name}: {details}")
    return counts

train_class_counts = print_class_distribution("Train distribution", train_labels)
print_class_distribution("Val distribution", val_labels)
print_class_distribution("Test distribution", test_labels)

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def audit_split_overlap(split_paths):
    """Audit leakage that can still be checked when synthetic data has no patient id."""
    name_sets = {name: {os.path.basename(p) for p in paths} for name, paths in split_paths.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = name_sets[left] & name_sets[right]
        print(f"[AUDIT] Same filenames {left}/{right}: {len(overlap)}")
        if overlap:
            print(f"[WARN] Possible split leakage, examples: {sorted(overlap)[:5]}")

    if AUDIT_EXACT_DUPLICATES:
        hash_sets = {
            name: {_sha256(path) for path in paths}
            for name, paths in split_paths.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = hash_sets[left] & hash_sets[right]
            print(f"[AUDIT] Exact duplicate content {left}/{right}: {len(overlap)}")
            if overlap:
                raise ValueError(f"Exact duplicate images found across {left}/{right} splits.")

audit_split_overlap({"train": train_paths, "val": val_paths, "test": test_paths})

if len(train_paths) == 0:
    raise ValueError(f"[LỖI CRITICAL] Không tìm thấy ảnh nào trong {TRAIN_DIR}. Hãy kiểm tra lại file zip của bạn xem cấu trúc thư mục có đúng là split_dataset/train/0,1,2,3,4 không!")
if len(val_paths) == 0 or len(test_paths) == 0:
    raise ValueError("Validation hoặc test đang rỗng; không thể chọn model và đánh giá đáng tin cậy.")

# [FIX] Shuffle toàn cục danh sách file trước khi đưa vào tf.data.Dataset
# Khắc phục lỗi dataset bị gom cụm theo từng class do đọc từ thư mục
indices = np.arange(len(train_paths))
np.random.shuffle(indices)
train_paths = train_paths[indices]
train_labels = train_labels[indices]

if USE_CLASS_WEIGHTS:
    safe_counts = np.maximum(train_class_counts.astype(np.float64), 1.0)
    class_weights = (safe_counts.max() / safe_counts) ** CLASS_WEIGHT_POWER
    class_weights = np.minimum(class_weights, MAX_CLASS_WEIGHT)
    class_weights = class_weights / np.average(class_weights, weights=safe_counts)
else:
    class_weights = np.ones(NUM_CLASSES, dtype=np.float64)
print("[DATA] Train class weights:", {i: round(float(w), 3) for i, w in enumerate(class_weights)})
train_sample_weights = class_weights[train_labels].astype(np.float32)

def load_and_preprocess_from_path(image_path, label, sample_weight=None):
    image_path = tf.cast(image_path, tf.string)
    img = tf.io.read_file(image_path)
    # Dùng decode_image thay vì decode_jpeg để tương thích cả PNG và JPEG trong dataset
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])  # Bắt buộc vì decode_image không tự set static shape
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    if sample_weight is None:
        return img, label
    return img, label, sample_weight

def create_dataset(paths, labels, batch_size, is_training=True, sample_weights=None):
    tensors = (paths, labels, sample_weights) if sample_weights is not None else (paths, labels)
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    if is_training:
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.repeat()
    
    # deterministic=False giúp xử lý đa luồng nhanh hơn, không quan tâm thứ tự trả về
    dataset = dataset.map(load_and_preprocess_from_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    
    # Sử dụng hàm ignore_errors mới của tf.data thay vì dùng thư viện experimental
    # Fail loudly on corrupt images; silently dropping them makes epoch accounting unreliable.
        
    # Thêm .cache() để lưu trữ dataset sau epoch 1 (RAM hoặc Disk), giúp các epoch sau siêu tốc
    # Nếu bị đầy RAM (như trường hợp của bạn), hãy TẮT lệnh này đi. Đọc từ ổ cứng Colab đã khá nhanh.
    # dataset = dataset.cache()
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(
    train_paths, train_labels, BATCH_SIZE,
    is_training=True, sample_weights=train_sample_weights
)
val_dataset   = create_dataset(val_paths,   val_labels,   BATCH_SIZE, is_training=False)
test_dataset  = create_dataset(test_paths,  test_labels,  BATCH_SIZE, is_training=False)

STEPS_PER_EPOCH = int(np.ceil(len(train_paths) / BATCH_SIZE))

# ============================================================
#  BƯỚC 4: CUSTOM METRIC CHO COHEN'S KAPPA (CORAL-compatible)
# ============================================================
class CohenKappaMetric(tf.keras.metrics.Metric):
    """
    Tính Toán Quadratic Weighted Kappa (QWK) trực tiếp trong lúc model.fit() 
    chạy validation. Tương thích với CORAL ordinal output (4 sigmoid neurons).
    """
    def __init__(self, num_classes=5, name='kappa', **kwargs):
        super(CohenKappaMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(
            name='conf_mtx', 
            shape=(num_classes, num_classes), 
            initializer='zeros', 
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # CORAL ordinal: sum(P(Y>k) > 0.5) → class prediction
        y_pred = tf.reduce_sum(tf.cast(y_pred > 0.5, tf.int32), axis=-1)
        y_pred = tf.reshape(y_pred, [-1])
        
        conf_mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        self.conf_mtx.assign_add(conf_mtx)

    def result(self):
        w = tf.zeros([self.num_classes, self.num_classes], dtype=tf.float32)
        indices = []
        updates = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                indices.append([i, j])
                updates.append(float((i - j) ** 2))
        w = tf.tensor_scatter_nd_update(w, indices, updates)
        w = w / float((self.num_classes - 1) ** 2)
        
        hist_true = tf.reduce_sum(self.conf_mtx, axis=1)
        hist_pred = tf.reduce_sum(self.conf_mtx, axis=0)
        
        e = tf.tensordot(hist_true, hist_pred, axes=0) / (tf.reduce_sum(self.conf_mtx) + tf.keras.backend.epsilon())
        
        num = tf.reduce_sum(w * self.conf_mtx)
        den = tf.reduce_sum(w * e)
        
        return 1.0 - (num / (den + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.conf_mtx.assign(tf.zeros((self.num_classes, self.num_classes), dtype=tf.float32))

# ============================================================
#  BƯỚC 5: CUTOUT LAYER (Data Augmentation nâng cao)
# ============================================================
class CutOutLayer(layers.Layer):
    """Random Erasing / CutOut — mask 1 vùng vuông ngẫu nhiên trên ảnh.
    Giúp model không phụ thuộc vào một vùng duy nhất, tăng generalization."""
    def __init__(self, mask_size_ratio=0.15, **kwargs):
        super().__init__(**kwargs)
        self.mask_size_ratio = mask_size_ratio

    def call(self, images, training=None):
        if not training:
            return images
        batch_size = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        mask_h = tf.cast(tf.cast(h, tf.float32) * self.mask_size_ratio, tf.int32)
        mask_w = tf.cast(tf.cast(w, tf.float32) * self.mask_size_ratio, tf.int32)

        # Random vị trí top-left cho mỗi ảnh trong batch
        top = tf.random.uniform([batch_size, 1, 1, 1], 0, h - mask_h, dtype=tf.int32)
        left = tf.random.uniform([batch_size, 1, 1, 1], 0, w - mask_w, dtype=tf.int32)

        # Tạo mask grid
        row_idx = tf.range(h)[tf.newaxis, :, tf.newaxis, tf.newaxis]
        col_idx = tf.range(w)[tf.newaxis, tf.newaxis, :, tf.newaxis]

        mask = ~((row_idx >= top) & (row_idx < top + mask_h) &
                 (col_idx >= left) & (col_idx < left + mask_w))
        mask = tf.cast(mask, images.dtype)
        return images * mask

    def get_config(self):
        config = super().get_config()
        config.update({"mask_size_ratio": self.mask_size_ratio})
        return config

# ============================================================
#  BƯỚC 5b: SERIALIZABLE CUSTOM LAYERS (thay thế Lambda)
# ============================================================
@tf.keras.utils.register_keras_serializable(package='Custom')
class PreprocessInputLayer(layers.Layer):
    """
    Layer chuẩn hóa đầu vào thay cho layers.Lambda(preprocess_input).
    Được đăng ký với @register_keras_serializable nên có thể load/save model đúng cách.
    """
    def __init__(self, model_name='EfficientNetB3', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def call(self, x):
        if self.model_name == 'EfficientNetB3':
            return effnet_preprocess(x)
        elif self.model_name == 'ResNet50':
            return resnet_preprocess(x)
        elif self.model_name == 'ConvNeXtTiny':
            return convnext_preprocess(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'model_name': self.model_name})
        return config


@tf.keras.utils.register_keras_serializable(package='Custom')
class GeMPoolingLayer(layers.Layer):
    """
    Generalized Mean Pooling (GeM) — thay thế layers.Lambda cho GeM pooling.
    GeM(x) = (mean(x^p))^(1/p). p > 1 nhấn mạnh vùng activation cao.
    Được đăng ký với @register_keras_serializable để đảm bảo serialization.
    """
    def __init__(self, p=3.0, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.maximum(x, 1e-6)
        return tf.pow(
            tf.reduce_mean(tf.pow(x, self.p), axis=[1, 2]),
            1.0 / self.p
        )

    def get_config(self):
        config = super().get_config()
        config.update({'p': self.p})
        return config


# ============================================================
#  BƯỚC 6: XÂY DỰNG MÔ HÌNH (CORAL Ordinal Classification)
# ============================================================
def build_model():
    # Data Augmentation cải thiện — thêm nhiều transform + CutOut
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom((-0.10, 0.10)),
        # Mức nhẹ giúp model bền hơn với khác biệt camera/chiếu sáng, kể cả ảnh raw.
        layers.RandomContrast(0.10),
        layers.RandomTranslation(0.02, 0.02),
        CutOutLayer(mask_size_ratio=0.0),
    ], name="data_augmentation")

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)

    if MODEL_NAME == 'EfficientNetB3':
        x = PreprocessInputLayer(model_name='EfficientNetB3', name="preprocess_input")(x)
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif MODEL_NAME == 'ResNet50':
        x = PreprocessInputLayer(model_name='ResNet50', name="preprocess_input")(x)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif MODEL_NAME == 'ConvNeXtTiny':
        x = PreprocessInputLayer(model_name='ConvNeXtTiny', name="preprocess_input")(x)
        base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("MODEL_NAME không hợp lệ.")

    base_model.trainable = False  
    x = base_model(x, training=False)

    # ---- HEAD MỚI: GeM Pooling + GELU ----
    # Generalized Mean Pooling (GeM) — tốt hơn GlobalAveragePooling cho fine-grained recognition
    # GeM(x) = (mean(x^p))^(1/p), với p > 1 sẽ nhấn mạnh vào vùng có activation cao
    x = GeMPoolingLayer(p=3.0, name="gem_pooling")(x)

    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, name="head_dense1")(x)
    x = layers.Activation('gelu', name="head_gelu1")(x)   # GELU thay ReLU — smoother gradient
    x = layers.Dropout(0.4, name="head_drop1")(x)
    x = layers.Dense(256, name="head_dense2")(x)
    x = layers.Activation('gelu', name="head_gelu2")(x)
    x = layers.Dropout(0.25, name="head_drop2")(x)

    # CORAL Ordinal Output: 4 sigmoid neurons — P(Y>0), P(Y>1), P(Y>2), P(Y>3)
    outputs = layers.Dense(NUM_CLASSES - 1, activation='sigmoid', dtype='float32', name="ordinal_output")(x)
    model = models.Model(inputs, outputs)
    return model, base_model

# ============================================================
#  BƯỚC 7: HUẤN LUYỆN MÔ HÌNH (STANDARD RUN)
# ============================================================
print(f"\n{'='*65}")
print(f"  BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH {MODEL_NAME} (CORAL Ordinal)")
print(f"{'='*65}")

# ----------------------------------------------------
# RESUME: Load checkpoint nếu đã tồn tại (tránh mất progress khi Colab crash)
# ----------------------------------------------------
checkpoint_path = os.path.join(MODELS_DIR, f"best_{EXPERIMENT_NAME}.keras")

if EVALUATE_ONLY and not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"EVALUATE_ONLY=True but checkpoint was not found: {checkpoint_path}"
    )

if os.path.exists(checkpoint_path):
    print(f"[RESUME] ⚡ Tìm thấy checkpoint tại: {checkpoint_path}")
    print(f"[RESUME] Đang load model để tiếp tục training...")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={
            'ordinal_loss': ordinal_loss,
            'CohenKappaMetric': CohenKappaMetric,
            'CutOutLayer': CutOutLayer,
            'PreprocessInputLayer': PreprocessInputLayer,
            'GeMPoolingLayer': GeMPoolingLayer,
        }
    )
    # Lấy lại base_model từ layer của model đã load (để dùng trong Phase 2)
    backbone_layer_name = {
        'EfficientNetB3': 'efficientnetb3',
        'ResNet50': 'resnet50',
        'ConvNeXtTiny': 'convnext_tiny',
    }[MODEL_NAME]
    base_model = model.get_layer(backbone_layer_name)
    print(f"[RESUME] Load model thành công! Bỏ qua Phase 1, tiến thẳng vào Phase 2.")
    phase1_history = None  # Không có phase 1 history khi resume
else:
    model, base_model = build_model()

    # ----------------------------------------------------
    # PHASE 1: WARMUP HEAD
    # ----------------------------------------------------
    print(f"\n[PHASE 1] TRAIN CLASSIFICATION HEAD (WARMUP)")
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4),
        loss=ordinal_loss,
        metrics=[CohenKappaMetric(num_classes=NUM_CLASSES)]
    )

    try:
        phase1_history = model.fit(
            train_dataset,
            epochs=EPOCHS_HEAD,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=val_dataset,
            verbose=1
        )
    except Exception as e:
        print(f"[ERROR] Phase 1 training failed: {e}")
        raise

    # Lưu history Phase 1 ngay sau khi xong — tránh mất log nếu crash ở Phase 2
    phase1_hist_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_phase1_history.pkl")
    with open(phase1_hist_path, 'wb') as f:
        pickle.dump(phase1_history.history, f)
    print(f"[DONE] Phase 1 history đã lưu: {phase1_hist_path}")

# ----------------------------------------------------
# PHASE 2: FINE-TUNING TOP LAYERS OF BACKBONE
# ----------------------------------------------------
print(f"\n[PHASE 2] FINE-TUNING TOP LAYERS OF BACKBONE")
base_model.trainable = True

# Fine-tune 30% cuối backbone. EfficientNet vẫn giữ phần đặc trưng cơ bản ổn định.
freeze_ratio = 0.7
num_layers = len(base_model.layers)
freeze_until = int(num_layers * freeze_ratio)

for layer in base_model.layers[:freeze_until]:
    layer.trainable = False
    
print(f"[INFO] Đã đóng băng {freeze_until}/{num_layers} lớp đầu tiên ({freeze_ratio*100:.0f}%). Chỉ train {num_layers - freeze_until} lớp cuối.")

# LR cố định 2e-5 + ReduceLROnPlateau (thay CosineDecay)
initial_lr = 1e-5

model.compile(
    optimizer=optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-4),
    loss=ordinal_loss,
    metrics=[CohenKappaMetric(num_classes=NUM_CLASSES)]
)

# Thư mục backup cho BackupAndRestore — giúp tự động resume epoch nếu crash giữa chừng
backup_dir = os.path.join(MODELS_DIR, f'backup_{EXPERIMENT_NAME}')
os.makedirs(backup_dir, exist_ok=True)

callbacks_list = [
    # ReduceLROnPlateau: giảm LR 50% khi val_kappa không cải thiện sau 3 epochs
    callbacks.ReduceLROnPlateau(
        monitor='val_kappa',
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_kappa', # Theo dõi QWK trên tập val
        mode='max',
        patience=8,          # EarlyStopping patience
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_kappa', # Lưu model có val_kappa cao nhất
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    # BackupAndRestore: tự động lưu trạng thái mỗi epoch, giúp resume khi Colab bị ngắt
    callbacks.BackupAndRestore(backup_dir=backup_dir)
]

phase2_history = None
if EVALUATE_ONLY:
    print("[EVALUATE_ONLY] Skipping Phase 2 training.")
else:
    try:
        phase2_history = model.fit(
            train_dataset,
            epochs=EPOCHS_TUNE,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=val_dataset,
            callbacks=callbacks_list,
            verbose=1
        )
    except Exception as e:
        print(f"[ERROR] Phase 2 training failed: {e}")
        raise

# Always evaluate the checkpoint selected by validation QWK, even when training
# reaches the final epoch without EarlyStopping being triggered.
model = tf.keras.models.load_model(
    checkpoint_path,
    custom_objects={
        'ordinal_loss': ordinal_loss,
        'CohenKappaMetric': CohenKappaMetric,
        'CutOutLayer': CutOutLayer,
        'PreprocessInputLayer': PreprocessInputLayer,
        'GeMPoolingLayer': GeMPoolingLayer,
    }
)
print(f"[CHECKPOINT] Loaded best validation model: {checkpoint_path}")

history_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_history.pkl")
if EVALUATE_ONLY:
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            full_history = pickle.load(f)
        print(f"[EVALUATE_ONLY] Loaded existing history: {history_path}")
    else:
        full_history = {}
        print(f"[EVALUATE_ONLY] History not found; continuing without it.")
else:
    # Merge history for both fresh and resumed training runs.
    full_history = {}
    if phase1_history is not None:
        for key in phase1_history.history.keys():
            p2_vals = phase2_history.history.get(key, [])
            full_history[key] = phase1_history.history[key] + p2_vals
    else:
        full_history = phase2_history.history

    with open(history_path, 'wb') as f:
        pickle.dump(full_history, f)

print(f"\n{'='*65}")
print(f"  HOÀN THÀNH HUẤN LUYỆN")
best_val_kappa = max(full_history.get('val_kappa', [0]))
print(f"  Best Val Kappa: {best_val_kappa:.4f}")
print(f"{'='*65}")
print(f"\n[DONE] Model lưu tại: {checkpoint_path}")
print(f"[DONE] Lịch sử train lưu tại: {history_path}")

# ============================================================
#  BƯỚC 9: TEST TIME AUGMENTATION (TTA)
# ============================================================
def predict_with_tta(model, dataset, n_augmentations=5):
    """
    Test Time Augmentation cho CORAL ordinal model.
    5 augmentations: Original, Horizontal flip, Slight brightness, Slight contrast, Scale.
    Average ordinal probabilities rồi convert → class.
    
    Args:
        model: Trained Keras model (CORAL ordinal output)
        dataset: tf.data.Dataset (batched, KHÔNG repeat)
        n_augmentations: Số phiên bản augmentation (tối đa 5)
    
    Returns:
        avg_preds: numpy array, shape (num_samples, NUM_CLASSES-1), averaged ordinal probabilities
        all_labels: numpy array, shape (num_samples,), true labels
    """
    tta_augmentations = [
        ("Original", lambda x: x),
        ("Horizontal Flip", lambda x: tf.image.flip_left_right(x)),
        # Brightness & Contrast không cần vì ảnh đã qua Ben Graham
        ("Scale (zoom)", lambda x: tf.image.resize(
            tf.image.central_crop(x, central_fraction=0.9),
            [IMG_SIZE, IMG_SIZE]
        )),
    ]
    
    selected_augs = tta_augmentations[:n_augmentations]
    
    all_preds = []
    all_labels = None
    
    for aug_idx, (aug_name, aug_fn) in enumerate(selected_augs):
        print(f"  [TTA] {aug_idx + 1}/{len(selected_augs)}: {aug_name}...")
        preds_batch = []
        labels_batch = []
        
        for images, labels in dataset:
            # Apply augmentation per-image trong batch
            augmented = tf.map_fn(aug_fn, images)
            preds = model(augmented, training=False)
            preds_batch.append(preds.numpy())
            labels_batch.append(labels.numpy())
        
        all_preds.append(np.concatenate(preds_batch, axis=0))
        if all_labels is None:
            all_labels = np.concatenate(labels_batch, axis=0)
    
    # Trung bình ordinal probabilities qua các augmentations
    avg_preds = np.mean(all_preds, axis=0)
    # CORAL cumulative probabilities must satisfy p(Y>0) >= ... >= p(Y>3).
    avg_preds = np.minimum.accumulate(avg_preds, axis=1)
    
    print(f"  [TTA] Done! Averaged {len(selected_augs)} augmentations.")
    return avg_preds, all_labels

def ordinal_probs_to_classes(probs, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return np.sum(probs > thresholds, axis=1).astype(np.int32)

def threshold_validation_metrics(y_true, probs, thresholds):
    classes = ordinal_probs_to_classes(probs, thresholds)
    return {
        "qwk": cohen_kappa_score(y_true, classes, weights='quadratic'),
        "macro_f1": f1_score(y_true, classes, average='macro', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, classes),
    }

def optimize_shared_ordinal_threshold(y_true, probs):
    """Tune only one shared threshold to reduce validation overfitting.

    The selection score keeps the ordinal objective (QWK) dominant while also
    preventing a high QWK from hiding a weak middle class.
    """
    candidates = []
    search_grid = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
    for value in search_grid:
        thresholds = np.full(NUM_CLASSES - 1, value, dtype=np.float32)
        metrics = threshold_validation_metrics(y_true, probs, thresholds)
        selection_score = 0.70 * metrics["qwk"] + 0.30 * metrics["macro_f1"]
        candidates.append((selection_score, metrics["qwk"], value, metrics))

    # Include the canonical CORAL threshold explicitly, even if grid settings change.
    default_thresholds = np.full(NUM_CLASSES - 1, 0.5, dtype=np.float32)
    default_metrics = threshold_validation_metrics(y_true, probs, default_thresholds)
    default_score = 0.70 * default_metrics["qwk"] + 0.30 * default_metrics["macro_f1"]
    candidates.append((default_score, default_metrics["qwk"], 0.5, default_metrics))

    _, _, best_value, best_metrics = max(candidates, key=lambda item: (item[0], item[1]))
    return np.full(NUM_CLASSES - 1, best_value, dtype=np.float32), best_metrics

def optimize_separate_ordinal_thresholds(y_true, probs, max_passes=5):
    """
    Tối ưu riêng 4 threshold CORAL bằng coordinate descent.

    Threshold được chọn chỉ từ validation.
    Tuyệt đối không tối ưu bằng test set.
    """
    # Bắt đầu từ shared threshold tốt nhất để kết quả không tệ hơn baseline.
    best_thresholds, best_metrics = optimize_shared_ordinal_threshold(
        y_true, probs
    )

    def selection_score(metrics):
        return 0.70 * metrics["qwk"] + 0.30 * metrics["macro_f1"]

    best_score = selection_score(best_metrics)
    search_grid = np.linspace(
        THRESHOLD_MIN,
        THRESHOLD_MAX,
        THRESHOLD_STEPS
    )

    print(
        "[THRESHOLD] Starting separate optimization from:",
        best_thresholds
    )

    for pass_index in range(max_passes):
        pass_improved = False

        for boundary_index in range(NUM_CLASSES - 1):
            boundary_best_thresholds = best_thresholds.copy()
            boundary_best_metrics = best_metrics
            boundary_best_score = best_score

            for value in search_grid:
                candidate_thresholds = best_thresholds.copy()
                candidate_thresholds[boundary_index] = value

                metrics = threshold_validation_metrics(
                    y_true,
                    probs,
                    candidate_thresholds
                )
                score = selection_score(metrics)

                is_better = (
                    score > boundary_best_score + 1e-8
                    or (
                        abs(score - boundary_best_score) <= 1e-8
                        and metrics["qwk"]
                        > boundary_best_metrics["qwk"]
                    )
                )

                if is_better:
                    boundary_best_thresholds = candidate_thresholds
                    boundary_best_metrics = metrics
                    boundary_best_score = score

            if boundary_best_score > best_score + 1e-8:
                best_thresholds = boundary_best_thresholds
                best_metrics = boundary_best_metrics
                best_score = boundary_best_score
                pass_improved = True

        print(
            f"[THRESHOLD] Pass {pass_index + 1}: "
            f"thresholds={np.round(best_thresholds, 3)}, "
            f"QWK={best_metrics['qwk']:.4f}, "
            f"Macro F1={best_metrics['macro_f1']:.4f}"
        )

        if not pass_improved:
            break

    return best_thresholds.astype(np.float32), best_metrics

# ============================================================
#  BƯỚC 10: ĐÁNH GIÁ VỚI TTA + CONFUSION MATRIX + AUC
# ============================================================
print(f"\n{'='*65}")
print(f"  ĐÁNH GIÁ MODEL VỚI TTA + CONFUSION MATRIX + AUC")
print(f"{'='*65}")

tta_preds, tta_labels = predict_with_tta(
    model, val_dataset, n_augmentations=TTA_AUGMENTATIONS
)

# Save raw validation outputs so thresholds can be retuned without rerunning TTA.
val_probs_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_val_ordinal_probs.npy")
val_labels_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_val_labels.npy")
np.save(val_probs_path, tta_preds)
np.save(val_labels_path, tta_labels)
print(f"[THRESHOLD] Validation probabilities saved at: {val_probs_path}")
print(f"[THRESHOLD] Validation labels saved at: {val_labels_path}")

# Tune thresholds strictly on validation, then freeze them for test inference.
default_thresholds = np.full(NUM_CLASSES - 1, 0.5, dtype=np.float32)
default_val_metrics = threshold_validation_metrics(tta_labels, tta_preds, default_thresholds)
shared_thresholds, shared_val_metrics = optimize_shared_ordinal_threshold(tta_labels, tta_preds)
ordinal_thresholds, optimized_val_metrics = optimize_separate_ordinal_thresholds(
    tta_labels, tta_preds, max_passes=5
)
optimized_val_kappa = optimized_val_metrics["qwk"]
tta_classes = ordinal_probs_to_classes(tta_preds, ordinal_thresholds)
threshold_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_ordinal_thresholds.npy")
np.save(threshold_path, ordinal_thresholds)
print(f"[THRESHOLD] Default 0.5 metrics: {json.dumps(default_val_metrics, indent=2)}")
print(f"[THRESHOLD] Best shared thresholds: {np.round(shared_thresholds, 3)}")
print(f"[THRESHOLD] Shared metrics: {json.dumps(shared_val_metrics, indent=2)}")
print(f"[THRESHOLD] Separate thresholds: {np.round(ordinal_thresholds, 3)}")
print(f"[THRESHOLD] Separate metrics: {json.dumps(optimized_val_metrics, indent=2)}")
print(f"[THRESHOLD] Saved at: {threshold_path}")

# --- Quadratic Weighted Kappa ---
tta_kappa = cohen_kappa_score(tta_labels, tta_classes, weights='quadratic')
print(f"\n  Val Kappa (no TTA):   {best_val_kappa:.4f}")
print(f"  Val Kappa (with TTA): {tta_kappa:.4f}")
print(f"  Improvement:          {tta_kappa - best_val_kappa:+.4f}")

# --- Confusion Matrix ---
print(f"\n{'='*65}")
print(f"  CONFUSION MATRIX")
print(f"{'='*65}")
cm = confusion_matrix(tta_labels, tta_classes, labels=list(range(NUM_CLASSES)))
# Header
class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
header = "True\\Pred  " + "  ".join(f"{name:>8}" for name in class_names)
print(header)
print("-" * len(header))
for i in range(NUM_CLASSES):
    row = f"Class {i}    " + "  ".join(f"{cm[i, j]:>8d}" for j in range(NUM_CLASSES))
    print(row)
print()

# Per-class accuracy
print("  Per-class Accuracy:")
for i in range(NUM_CLASSES):
    class_total = cm[i].sum()
    class_correct = cm[i, i]
    acc = class_correct / class_total if class_total > 0 else 0.0
    print(f"    Class {i}: {class_correct}/{class_total} = {acc:.2%}")

overall_acc = np.trace(cm) / cm.sum()
val_balanced_acc = balanced_accuracy_score(tta_labels, tta_classes)
val_macro_f1 = f1_score(tta_labels, tta_classes, average='macro', zero_division=0)
print(f"  Overall Accuracy: {overall_acc:.2%}")
print(f"  Balanced Accuracy: {val_balanced_acc:.2%}")
print(f"  Macro F1: {val_macro_f1:.4f}")

# --- AUC (One-vs-Rest) ---
print(f"\n{'='*65}")
print(f"  AUC (One-vs-Rest)")
print(f"{'='*65}")

# Chuyển ordinal probabilities → softmax-like probabilities cho AUC
# P(Y=0) = 1 - P(Y>0)
# P(Y=k) = P(Y>k-1) - P(Y>k) for k in [1, NUM_CLASSES-2]
# P(Y=NUM_CLASSES-1) = P(Y>NUM_CLASSES-2)
ordinal_probs_padded = np.concatenate([
    np.ones((len(tta_preds), 1)),   # P(Y >= 0) = 1.0
    tta_preds,                       # P(Y > 0), P(Y > 1), P(Y > 2), P(Y > 3)
    np.zeros((len(tta_preds), 1))   # P(Y > NUM_CLASSES-1) = 0.0
], axis=1)

# P(Y=k) = P(Y >= k) - P(Y >= k+1) = P(Y > k-1) - P(Y > k)
class_probs = np.diff(-ordinal_probs_padded, axis=1)  # = padded[:, :-1] - padded[:, 1:]
class_probs = np.clip(class_probs, 0, 1)  # Đảm bảo không âm

# One-hot encode true labels cho AUC
y_true_onehot = np.eye(NUM_CLASSES)[tta_labels.astype(int)]

macro_auc = float('nan')
try:
    # Macro AUC (trung bình AUC của từng class)
    macro_auc = roc_auc_score(y_true_onehot, class_probs, multi_class='ovr', average='macro')
    print(f"  Macro AUC (OvR): {macro_auc:.4f}")
    
    # Per-class AUC
    for i in range(NUM_CLASSES):
        try:
            class_auc = roc_auc_score(y_true_onehot[:, i], class_probs[:, i])
            print(f"    Class {i} AUC: {class_auc:.4f}")
        except ValueError:
            print(f"    Class {i} AUC: N/A (chỉ có 1 class trong tập val)")
except ValueError as e:
    print(f"  [WARN] Không thể tính AUC: {e}")

# --- Lưu confusion matrix ---
cm_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_confusion_matrix.npy")
np.save(cm_path, cm)
print(f"\n[DONE] Confusion matrix lưu tại: {cm_path}")

print(f"{'='*65}")
print(f"  TỔNG KẾT (VALIDATION)")
print(f"  Val Kappa (TTA):  {tta_kappa:.4f}")
print(f"  Macro AUC (OvR):  {macro_auc:.4f}")
print(f"  Overall Accuracy:  {overall_acc:.2%}")
print(f"{'='*65}")

# ==============================
#  BƯỚC 11: ĐÁNH GIÁ TRÊN TẬP TEST
# ==============================
print("\n" + "="*65)
print(f"   ĐÁNH GIÁ TRÊN TẬP TEST (TTA x{TTA_AUGMENTATIONS})")
print("="*65)
# Dùng đúng cùng pipeline inference đã dùng để hiệu chỉnh threshold trên validation.
test_preds, test_labels = predict_with_tta(
    model, test_dataset, n_augmentations=TTA_AUGMENTATIONS
)

# Save raw test outputs for reproducible offline evaluation. Test labels are
# never used to select thresholds.
test_probs_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_test_ordinal_probs.npy")
test_labels_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_test_labels.npy")
np.save(test_probs_path, test_preds)
np.save(test_labels_path, test_labels)
print(f"[TEST] Ordinal probabilities saved at: {test_probs_path}")
print(f"[TEST] Labels saved at: {test_labels_path}")

# Chuyển ordinal → class
test_classes = ordinal_probs_to_classes(test_preds, ordinal_thresholds)
# Confusion matrix cho test
test_cm = confusion_matrix(test_labels, test_classes, labels=list(range(NUM_CLASSES)))
print("\nTEST CONFUSION MATRIX")
class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
header = "True\\Pred  " + "  ".join(f"{name:>8}" for name in class_names)
print(header)
print("-" * len(header))
for i in range(NUM_CLASSES):
    row = f"Class {i}    " + "  ".join(f"{test_cm[i, j]:>8d}" for j in range(NUM_CLASSES))
    print(row)
# Per‑class accuracy on test
print("\n  Per‑class Accuracy (Test):")
worst_class = None
worst_acc = 1.0
for i in range(NUM_CLASSES):
    total = test_cm[i].sum()
    correct = test_cm[i, i]
    acc = correct / total if total > 0 else 0.0
    print(f"    Class {i}: {correct}/{total} = {acc:.2%}")
    if acc < worst_acc:
        worst_acc = acc
        worst_class = i
print(f"\n  → Mức DR mà mô hình yếu nhất: Class {worst_class} (accuracy {worst_acc:.2%})")
test_overall_acc = np.trace(test_cm) / test_cm.sum()
test_kappa = cohen_kappa_score(test_labels, test_classes, weights='quadratic')
test_balanced_acc = balanced_accuracy_score(test_labels, test_classes)
print(f"  Overall Accuracy (Test): {test_overall_acc:.2%}")
print(f"  Balanced Accuracy (Test): {test_balanced_acc:.2%}")
print(f"  Quadratic Weighted Kappa (Test): {test_kappa:.4f}")
# ------------------------------------------------------------------
#  Đánh giá thêm: Precision, Recall, F1 (macro) trên tập test
try:
    macro_precision = precision_score(test_labels, test_classes, average='macro', zero_division=0)
    macro_recall = recall_score(test_labels, test_classes, average='macro', zero_division=0)
    macro_f1 = f1_score(test_labels, test_classes, average='macro', zero_division=0)
    print(f"  Macro Precision (Test): {macro_precision:.4f}")
    print(f"  Macro Recall (Test):    {macro_recall:.4f}")
    print(f"  Macro F1-score (Test):  {macro_f1:.4f}")
except Exception as e:
    print(f"  [WARN] Không thể tính Precision/Recall/F1: {e}")

# Calculate the same ordinal BCE used during training from test predictions.
test_cumulative_labels = (
    test_labels.reshape(-1, 1) >= np.arange(1, NUM_CLASSES).reshape(1, -1)
).astype(np.float32)
test_cumulative_labels = (
    test_cumulative_labels * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
)
clipped_test_preds = np.clip(test_preds.astype(np.float32), 1e-7, 1.0 - 1e-7)
test_loss = float(np.mean(
    -(test_cumulative_labels * np.log(clipped_test_preds)
      + (1.0 - test_cumulative_labels) * np.log(1.0 - clipped_test_preds))
))

class_display_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
per_class_recall = {}
for class_index, class_name in enumerate(class_display_names):
    class_total = int(test_cm[class_index].sum())
    per_class_recall[class_name] = (
        float(test_cm[class_index, class_index] / class_total)
        if class_total > 0 else 0.0
    )

test_summary = {
    "loss": test_loss,
    "accuracy": float(test_overall_acc),
    "macro_f1": float(macro_f1) if 'macro_f1' in locals() else None,
    "balanced_accuracy": float(test_balanced_acc),
    "qwk": float(test_kappa),
    "per_class_recall": per_class_recall,
}
# Lưu confusion matrix test
test_cm_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_test_confusion_matrix.npy")
np.save(test_cm_path, test_cm)
print(f"[DONE] Test confusion matrix saved at: {test_cm_path}")
# Lưu toàn bộ metrics test vào pickle
test_metrics = {
    "test_confusion_matrix_path": test_cm_path,
    "test_macro_precision": macro_precision if 'macro_precision' in locals() else None,
    "test_macro_recall": macro_recall if 'macro_recall' in locals() else None,
    "test_macro_f1": macro_f1 if 'macro_f1' in locals() else None,
    "test_worst_class": worst_class,
    "test_worst_accuracy": worst_acc,
    "test_overall_accuracy": test_overall_acc,
    "test_balanced_accuracy": test_balanced_acc,
    "test_quadratic_weighted_kappa": test_kappa,
    "ordinal_thresholds": ordinal_thresholds.tolist(),
    "tta_augmentations": TTA_AUGMENTATIONS,
    "validation_default_threshold_metrics": default_val_metrics,
    "validation_optimized_threshold_metrics": optimized_val_metrics,
}
test_history_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_test_history.pkl")
with open(test_history_path, 'wb') as f:
    pickle.dump(test_metrics, f)
print(f"[DONE] Test metrics saved at: {test_history_path}")

test_summary_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_test_summary.json")
with open(test_summary_path, 'w', encoding='utf-8') as f:
    json.dump(test_summary, f, ensure_ascii=False, indent=2)

print("\n" + "="*65)
print("  FINAL TEST SUMMARY")
print("="*65)
print(json.dumps(test_summary, ensure_ascii=False, indent=2))
print(f"[DONE] Test summary JSON saved at: {test_summary_path}")
print("="*65)

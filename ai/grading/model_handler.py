import os
import numpy as np
import tensorflow as tf

# Mapping classes dựa trên thang chuẩn ICDR
CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR"
]

def build_efficientnet_b3(
    input_shape: tuple = (300, 300, 3),
    num_classes: int = 5,
    weights: str = None,
    training_base: bool = False,
    include_augmentation: bool = True,
    use_dual_pooling: bool = False,
):
    """
    Xây dựng kiến trúc EfficientNetB3 chuẩn cho classification 5 lớp DR grade.
    Khớp cấu trúc lớp với checkpoint best_EfficientNetB3_rgb_crop_v1.keras.
    """
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

    inputs = layers.Input(shape=input_shape)
    x = inputs

    if include_augmentation:
        data_augmentation = models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom((-0.1, 0.1)),
        ], name="data_augmentation")
        x = data_augmentation(x)

    x = layers.Lambda(effnet_preprocess, name="preprocess_input")(x)
    
    base_model = EfficientNetB3(weights=weights, include_top=False, input_shape=input_shape)
    x = base_model(x, training=training_base)

    if use_dual_pooling:
        avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        max_pool = layers.GlobalMaxPooling2D(name="max_pool")(x)
        x = layers.Concatenate(name="dual_pool")([avg_pool, max_pool])
    else:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', name="head_dense1")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', name="head_dense2")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name="predictions")(x)
    return models.Model(inputs, outputs, name="DR_EfficientNetB3_Grading")


class DRModelHandler:
    def __init__(self, model_path: str):
        """
        Khởi tạo và load model TensorFlow/Keras (EfficientNet-B3) vào RAM.
        """
        self.model_path = model_path
        self.model = None
        self.input_size = (300, 300) # EfficientNetB3 dùng size 300
        self.model_version = "efficientnet_b3_v1.0"
        
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) == 0:
            print(f"[!] CẢNH BÁO: Không tìm thấy file model tại {self.model_path}")
            print("[!] Vui lòng copy file model (.keras) của bạn vào đường dẫn này.")
            return
            
        print(f"[*] Đang load model từ {self.model_path}...")
        try:
            self.model = build_efficientnet_b3(
                input_shape=(300, 300, 3),
                num_classes=5,
                weights=None,
                training_base=False,
                include_augmentation=True,
                use_dual_pooling=False,
            )
            self.model.load_weights(self.model_path)
            print("[v] Load model thành công (Single Pooling)!")
            return
        except Exception:
            pass

        try:
            self.model = build_efficientnet_b3(
                input_shape=(300, 300, 3),
                num_classes=5,
                weights=None,
                training_base=False,
                include_augmentation=True,
                use_dual_pooling=True,
            )
            self.model.load_weights(self.model_path)
            print("[v] Load model thành công (Dual Pooling)!")
        except Exception as e:
            print(f"[x] Lỗi khi load model: {e}")
            
    def predict(self, preprocessed_img: np.ndarray):
        """
        Thực hiện dự đoán dựa trên ảnh đã tiền xử lý.
        preprocessed_img: numpy array từ OpenCV (BGR)
        """
        if self.model is None:
            raise ValueError("Model chưa được load. Vui lòng kiểm tra lại file .h5.")
            
        import cv2
        
        # Match the input size used by this checkpoint.
        img_resized = cv2.resize(preprocessed_img, self.input_size)
        
        # Chuyển BGR (OpenCV mặc định) sang RGB (TensorFlow/Keras thường dùng)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Chuyển kiểu dữ liệu sang float32 và normalize nếu cần
        # Lưu ý: Các pre-trained models của tf.keras.applications.efficientnet 
        # đã tích hợp sẵn lớp Rescaling nội bộ, nhưng nếu lúc train bạn tự normalize (/255.0) 
        # thì bạn cần bật cờ normalize ở đây. Giả định model tự xử lý hoặc đã chuẩn hóa.
        img_tensor = img_rgb.astype('float32')
        
        # Keras requires a batch dimension.
        img_batch = np.expand_dims(img_tensor, axis=0)
        
        # Chạy inference
        predictions = self.model.predict(img_batch, verbose=0)
        
        # predictions thường là mảng 2 chiều [[prob0, prob1, prob2, prob3, prob4]]
        # Nếu mô hình trả về logit, hàm tính xác suất sẽ phải dùng Softmax, 
        # nhưng thông thường layer cuối đã có activation='softmax'.
        probs = predictions[0].tolist()
        
        # Đảm bảo tổng xác suất = 1 (tránh sai số float)
        probability_sum = sum(probs)
        if probability_sum <= 0:
            raise ValueError("Model returned invalid probabilities")
        probs = [float(p) / probability_sum for p in probs]
        
        # Lấy class có xác suất cao nhất
        predicted_class = int(np.argmax(probs))
        confidence = probs[predicted_class]
        
        # Format kết quả theo chuẩn api_contract_ai.md
        result = {
            "dr_grade": predicted_class,
            "dr_label": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 4),
            "probabilities": {
                "No DR": round(probs[0], 4),
                "Mild NPDR": round(probs[1], 4),
                "Moderate NPDR": round(probs[2], 4),
                "Severe NPDR": round(probs[3], 4),
                "Proliferative DR": round(probs[4], 4)
            },
            "model_version": self.model_version
        }
        
        return result

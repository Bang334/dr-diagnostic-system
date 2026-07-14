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
        if not os.path.exists(self.model_path):
            print(f"[!] CẢNH BÁO: Không tìm thấy file model tại {self.model_path}")
            print("[!] Vui lòng copy file model (.keras) của bạn vào đường dẫn này.")
            return
            
        print(f"[*] Đang load model từ {self.model_path}...")
        try:
            # KHẮC PHỤC LỖI KHÁC VERSION KERAS (quantization_config):
            # Tự build lại ĐÚNG y hệt kiến trúc model thay vì dùng load_model để tránh lỗi parse config.
            from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
            from tensorflow.keras import layers, models
            from tensorflow.keras.applications import EfficientNetB3
            
            data_augmentation = models.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom((-0.1, 0.1)),
            ], name="data_augmentation")

            inputs = layers.Input(shape=(300, 300, 3))
            x = data_augmentation(inputs)
            x = layers.Lambda(effnet_preprocess, name="preprocess_input")(x)
            
            base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(300, 300, 3))
            x = base_model(x, training=False)

            avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            max_pool = layers.GlobalMaxPooling2D(name="max_pool")(x)
            x = layers.Concatenate(name="dual_pool")([avg_pool, max_pool])

            x = layers.BatchNormalization()(x)
            x = layers.Dense(512, activation='relu', name="head_dense1")(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(128, activation='relu', name="head_dense2")(x)
            x = layers.Dropout(0.3)(x)

            outputs = layers.Dense(5, activation='softmax', dtype='float32', name="predictions")(x)
            self.model = models.Model(inputs, outputs)
            
            # Load trọng số vào khung kiến trúc đã dựng chuẩn
            self.model.load_weights(self.model_path)
            print("[v] Load model thành công!")
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
        
        # Đảm bảo kích thước đúng 224x224 như model yêu cầu
        img_resized = cv2.resize(preprocessed_img, self.input_size)
        
        # Chuyển BGR (OpenCV mặc định) sang RGB (TensorFlow/Keras thường dùng)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Chuyển kiểu dữ liệu sang float32 và normalize nếu cần
        # Lưu ý: Các pre-trained models của tf.keras.applications.efficientnet 
        # đã tích hợp sẵn lớp Rescaling nội bộ, nhưng nếu lúc train bạn tự normalize (/255.0) 
        # thì bạn cần bật cờ normalize ở đây. Giả định model tự xử lý hoặc đã chuẩn hóa.
        img_tensor = img_rgb.astype('float32')
        
        # Keras yêu cầu input có batch dimension (shape: 1, 224, 224, 3)
        img_batch = np.expand_dims(img_tensor, axis=0)
        
        # Chạy inference
        predictions = self.model.predict(img_batch)
        
        # predictions thường là mảng 2 chiều [[prob0, prob1, prob2, prob3, prob4]]
        # Nếu mô hình trả về logit, hàm tính xác suất sẽ phải dùng Softmax, 
        # nhưng thông thường layer cuối đã có activation='softmax'.
        probs = predictions[0].tolist()
        
        # Đảm bảo tổng xác suất = 1 (tránh sai số float)
        probs = [float(p) / sum(probs) for p in probs]
        
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

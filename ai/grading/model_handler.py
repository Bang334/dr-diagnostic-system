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
        self.input_size = (224, 224) # Kích thước người dùng cung cấp
        self.model_version = "efficientnet_b3_v1.0"
        
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[!] CẢNH BÁO: Không tìm thấy file model tại {self.model_path}")
            print("[!] Vui lòng copy file model (.keras) của bạn vào đường dẫn này.")
            return
            
        print(f"[*] Đang load model từ {self.model_path}...")
        try:
            # Khởi tạo kiến trúc mạng y hệt config gốc để bypass lỗi Keras 3
            base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_shape=(224, 224, 3))
            
            self.model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
            
            # Chỉ load weights thay vì load toàn bộ model để tránh lỗi deserialization
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

import os
import numpy as np
import tensorflow as tf

class CutOutLayer(tf.keras.layers.Layer):
    """Random Erasing / CutOut — mask 1 vùng vuông ngẫu nhiên trên ảnh."""
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

        top = tf.random.uniform([batch_size, 1, 1, 1], 0, h - mask_h, dtype=tf.int32)
        left = tf.random.uniform([batch_size, 1, 1, 1], 0, w - mask_w, dtype=tf.int32)

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
        self.model_version = "efficientnet_b3_optimized_v1.0"
        
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
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom((-0.15, 0.15)),
                layers.RandomContrast(0.15),
                layers.RandomBrightness(0.08),
                layers.RandomTranslation(0.05, 0.05),
                CutOutLayer(mask_size_ratio=0.1),
            ], name="data_augmentation")

            inputs = layers.Input(shape=(300, 300, 3))
            x = data_augmentation(inputs)
            x = layers.Lambda(effnet_preprocess, name="preprocess_input")(x)
            
            base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(300, 300, 3))
            x = base_model(x, training=False)

            # ---- HEAD MỚI: GeM Pooling + GELU ----
            gem_p = 3.0
            x = layers.Lambda(
                lambda feat: tf.pow(
                    tf.reduce_mean(tf.pow(tf.maximum(tf.cast(feat, tf.float32), 1e-6), gem_p), axis=[1, 2]),
                    1.0 / gem_p
                ),
                name="gem_pooling"
            )(x)

            x = layers.BatchNormalization(name="head_bn")(x)
            x = layers.Dense(512, name="head_dense1")(x)
            x = layers.Activation('gelu', name="head_gelu1")(x)
            x = layers.Dropout(0.4, name="head_drop1")(x)
            x = layers.Dense(256, name="head_dense2")(x)
            x = layers.Activation('gelu', name="head_gelu2")(x)
            x = layers.Dropout(0.25, name="head_drop2")(x)

            # CORAL Ordinal Output: 4 sigmoid neurons
            outputs = layers.Dense(4, activation='sigmoid', dtype='float32', name="ordinal_output")(x)
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
        
        # CORAL Ordinal: Output là 4 xác suất tích lũy [P(Y>0), P(Y>1), P(Y>2), P(Y>3)]
        ordinal_probs = predictions[0].tolist()
        
        # Lấy class theo nguyên tắc Ordinal (Đếm số lượng P(Y>k) > 0.5)
        ordinal_predicted_class = sum(1 for p in ordinal_probs if p > 0.5)
        
        # Tính xác suất cho từng class rời rạc (One-vs-Rest / Marginal probability)
        p0 = max(0.0, 1.0 - ordinal_probs[0])
        p1 = max(0.0, ordinal_probs[0] - ordinal_probs[1])
        p2 = max(0.0, ordinal_probs[1] - ordinal_probs[2])
        p3 = max(0.0, ordinal_probs[2] - ordinal_probs[3])
        p4 = max(0.0, ordinal_probs[3])
        
        probs = [p0, p1, p2, p3, p4]
        
        # Chuẩn hóa lại tổng = 1 (tránh sai số float)
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [float(p) / total_prob for p in probs]
        else:
            probs = [1.0 if i == ordinal_predicted_class else 0.0 for i in range(5)]
            
        # Lấy class có xác suất rời rạc cao nhất (Argmax) để tối đa hoá Accuracy
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

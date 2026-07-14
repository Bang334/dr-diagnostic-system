import tensorflow as tf

import os

def test_load_weights():
    # 1. Khởi tạo kiến trúc mạng y hệt config.json
    base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_shape=(224, 224, 3))
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # 2. Load weights từ file .keras
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, 'ai', 'weights', 'dr_grading_model.keras')
    print(f"Loading weights from {model_path}...")
    model.load_weights(model_path)
    print("SUCCESS!")

test_load_weights()

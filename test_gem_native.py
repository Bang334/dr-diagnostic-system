import tensorflow as tf
import os

# Define GeMPoolingLayer without weight variables
@tf.keras.utils.register_keras_serializable(package="Custom", name="GeMPoolingLayer")
class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p = float(p)
        self.eps = float(eps)

    def call(self, inputs, training=None):
        x = tf.clip_by_value(inputs, self.eps, tf.float32.max)
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        x = tf.pow(x, 1.0 / self.p)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p, "eps": self.eps})
        return config

# Dummy custom objects for loading
custom_objs = {
    "GeMPoolingLayer": GeMPoolingLayer,
    "CutOutLayer": tf.keras.layers.Layer, # dummy
    "PreprocessInputLayer": tf.keras.layers.Layer, # dummy
    "ordinal_loss": lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true, y_pred),
    "CohenKappaMetric": tf.keras.metrics.Metric, # dummy
}

model_path = r"E:\HocTap\DoAnTT\best_EfficientNetB3_rgb_crop_v1.keras"
if os.path.exists(model_path):
    print("Testing native load_model...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
        print("SUCCESS! Model loaded native with 0 custom weight errors.")
        print("Total variables:", len(model.variables))
    except Exception as e:
        print("FAILED native load_model:", e)
else:
    print("Model file not found for test.")

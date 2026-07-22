"""Serializable objects required by the teacher model.

TensorFlow is intentionally isolated in this module so ordinal decoding and
pseudo-label selection remain importable in lightweight test environments.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


NUM_CLASSES = 5
LABEL_SMOOTHING = 0.05


@tf.keras.utils.register_keras_serializable(package="Custom")
class CutOutLayer(tf.keras.layers.Layer):
    def __init__(self, mask_size_ratio: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.mask_size_ratio = float(mask_size_ratio)

    def call(self, images, training=None):
        if not training or self.mask_size_ratio <= 0:
            return images
        shape = tf.shape(images)
        batch_size, height, width = shape[0], shape[1], shape[2]
        mask_height = tf.cast(
            tf.cast(height, tf.float32) * self.mask_size_ratio, tf.int32
        )
        mask_width = tf.cast(
            tf.cast(width, tf.float32) * self.mask_size_ratio, tf.int32
        )
        top = tf.random.uniform(
            [batch_size, 1, 1, 1], 0, height - mask_height, dtype=tf.int32
        )
        left = tf.random.uniform(
            [batch_size, 1, 1, 1], 0, width - mask_width, dtype=tf.int32
        )
        rows = tf.range(height)[tf.newaxis, :, tf.newaxis, tf.newaxis]
        columns = tf.range(width)[tf.newaxis, tf.newaxis, :, tf.newaxis]
        mask = ~(
            (rows >= top)
            & (rows < top + mask_height)
            & (columns >= left)
            & (columns < left + mask_width)
        )
        return images * tf.cast(mask, images.dtype)

    def get_config(self):
        return {**super().get_config(), "mask_size_ratio": self.mask_size_ratio}


@tf.keras.utils.register_keras_serializable(package="Custom")
class PreprocessInputLayer(tf.keras.layers.Layer):
    def __init__(self, model_name: str = "EfficientNetB3", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def call(self, inputs):
        if self.model_name == "EfficientNetB3":
            return preprocess_input(inputs)
        return inputs

    def get_config(self):
        return {**super().get_config(), "model_name": self.model_name}


@tf.keras.utils.register_keras_serializable(package="Custom")
class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.p = float(p)

    def call(self, inputs):
        values = tf.maximum(tf.cast(inputs, tf.float32), 1e-6)
        return tf.pow(
            tf.reduce_mean(tf.pow(values, self.p), axis=[1, 2]), 1.0 / self.p
        )

    def get_config(self):
        return {**super().get_config(), "p": self.p}


@tf.keras.utils.register_keras_serializable(package="Custom")
def ordinal_loss(y_true, y_pred):
    labels = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
    boundaries = tf.cast(tf.range(1, NUM_CLASSES), tf.float32)
    cumulative = tf.cast(labels >= boundaries, tf.float32)
    cumulative = cumulative * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
    return tf.keras.losses.binary_crossentropy(cumulative, y_pred)


@tf.keras.utils.register_keras_serializable(package="Custom")
class CohenKappaMetric(tf.keras.metrics.Metric):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        thresholds=(0.55, 0.50, 0.435, 0.31),
        name="kappa",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.thresholds = tuple(float(value) for value in thresholds)
        if len(self.thresholds) != self.num_classes - 1:
            raise ValueError("CohenKappaMetric requires one threshold per boundary")
        self.confusion = self.add_weight(
            name="conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        truth = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        thresholds = tf.constant(self.thresholds, dtype=y_pred.dtype)
        predicted = tf.reduce_sum(tf.cast(y_pred > thresholds, tf.int32), axis=-1)
        matrix = tf.math.confusion_matrix(
            truth, predicted, num_classes=self.num_classes, dtype=tf.float32
        )
        if sample_weight is not None:
            # Keras metrics receive per-image weights; weighting a pre-aggregated
            # matrix is ambiguous, so training uses the unweighted validation QWK.
            sample_weight = None
        self.confusion.assign_add(matrix)

    def result(self):
        indices = tf.range(self.num_classes, dtype=tf.float32)
        weights = tf.square(indices[:, None] - indices[None, :])
        weights /= float((self.num_classes - 1) ** 2)
        true_hist = tf.reduce_sum(self.confusion, axis=1)
        pred_hist = tf.reduce_sum(self.confusion, axis=0)
        expected = tf.tensordot(true_hist, pred_hist, axes=0)
        expected /= tf.reduce_sum(self.confusion) + tf.keras.backend.epsilon()
        numerator = tf.reduce_sum(weights * self.confusion)
        denominator = tf.reduce_sum(weights * expected)
        return 1.0 - numerator / (denominator + tf.keras.backend.epsilon())

    def reset_state(self):
        self.confusion.assign(tf.zeros_like(self.confusion))

    def get_config(self):
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "thresholds": list(self.thresholds),
        }


@tf.keras.utils.register_keras_serializable(package="Custom")
class OrdinalAccuracy(tf.keras.metrics.Metric):
    def __init__(
        self,
        thresholds=(0.55, 0.50, 0.435, 0.31),
        name="ordinal_accuracy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        truth = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        thresholds = tf.constant(self.thresholds, dtype=y_pred.dtype)
        predicted = tf.reduce_sum(tf.cast(y_pred > thresholds, tf.int32), axis=-1)
        matches = tf.cast(tf.equal(truth, predicted), self.dtype)
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
            matches *= weights
            count = tf.reduce_sum(weights)
        else:
            count = tf.cast(tf.size(matches), self.dtype)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(count)

    def result(self):
        return self.correct / (self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.correct.assign(0)
        self.total.assign(0)

    def get_config(self):
        return {**super().get_config(), "thresholds": list(self.thresholds)}


@tf.keras.utils.register_keras_serializable(package="Custom")
class OrdinalMeanAbsoluteError(tf.keras.metrics.Metric):
    def __init__(
        self,
        thresholds=(0.55, 0.50, 0.435, 0.31),
        name="ordinal_mae",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.error = self.add_weight(name="error", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        truth = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        thresholds = tf.constant(self.thresholds, dtype=y_pred.dtype)
        predicted = tf.cast(
            tf.reduce_sum(tf.cast(y_pred > thresholds, tf.int32), axis=-1),
            tf.float32,
        )
        errors = tf.abs(truth - predicted)
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            errors *= weights
            count = tf.reduce_sum(weights)
        else:
            count = tf.cast(tf.size(errors), tf.float32)
        self.error.assign_add(tf.reduce_sum(errors))
        self.total.assign_add(count)

    def result(self):
        return self.error / (self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.error.assign(0)
        self.total.assign(0)

    def get_config(self):
        return {**super().get_config(), "thresholds": list(self.thresholds)}


def get_custom_objects() -> dict[str, object]:
    return {
        "CutOutLayer": CutOutLayer,
        "Custom>CutOutLayer": CutOutLayer,
        "PreprocessInputLayer": PreprocessInputLayer,
        "Custom>PreprocessInputLayer": PreprocessInputLayer,
        "GeMPoolingLayer": GeMPoolingLayer,
        "Custom>GeMPoolingLayer": GeMPoolingLayer,
        "ordinal_loss": ordinal_loss,
        "Custom>ordinal_loss": ordinal_loss,
        "CohenKappaMetric": CohenKappaMetric,
        "Custom>CohenKappaMetric": CohenKappaMetric,
        "OrdinalAccuracy": OrdinalAccuracy,
        "Custom>OrdinalAccuracy": OrdinalAccuracy,
        "OrdinalMeanAbsoluteError": OrdinalMeanAbsoluteError,
        "Custom>OrdinalMeanAbsoluteError": OrdinalMeanAbsoluteError,
        "preprocess_input": preprocess_input,
    }

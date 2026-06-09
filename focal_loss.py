"""
focal_loss.py  —  Focal Loss compatible with TF 2.x / Keras 2.x and 3.x
=========================================================================
Uses keras.utils.register_keras_serializable which works across versions.
"""

import tensorflow as tf
import keras

@keras.utils.register_keras_serializable(package="custom")
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss as a Keras Loss class — fully serializable.
    This approach works with all TF/Keras versions.

    gamma: focusing parameter — higher = more focus on hard samples
    alpha: class balance factor
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        fl     = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config
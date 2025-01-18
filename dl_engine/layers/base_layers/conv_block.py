"""conv_block.py
"""

import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int = 3, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.norm = tf.keras.layers.BatchNormalization()
        if activation is not None:
            self.activation = tf.keras.layers.Activation(activation)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.norm(self.conv(x))
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x

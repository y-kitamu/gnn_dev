"""base.py
"""

import keras
import tensorflow as tf


class BaseNetwork(keras.Layer):
    @property
    def output_keys(self) -> list[str]:
        raise NotImplementedError

    def call(self, *args) -> dict[str, tf.Tensor]:
        raise NotImplementedError

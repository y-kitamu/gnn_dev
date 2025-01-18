"""sandbox.py
"""

import keras
import numpy as np
import tensorflow as tf

from ..base import BaseDataloader


class MnistDataloader(BaseDataloader):
    class Params(BaseDataloader.Params):
        batch_size: int = 32

    def __init__(self, params: Params, is_train: bool):
        self.params = params
        self.x, self.y = self.get_data(is_train)
        self.iterator = iter(
            tf.data.Dataset.from_tensor_slices((self.x, self.y))
            .shuffle(len(self.x))
            .batch(params.batch_size)
            .repeat()
        )

    @property
    def steps_per_epoch(self):
        return len(self.x) // self.params.batch_size

    @property
    def output_keys(self) -> list[str]:
        return ["input", "y_true"]

    @property
    def output_shape(self) -> list[int]:
        return [28, 28, 1]

    def get_next(self):
        x, y = next(self.iterator)
        return {"input": x, "y_true": y}

    def get_data(self, is_train: bool):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x = x_train if is_train else x_test
        y = y_train if is_train else y_test
        y = np.identity(10)[y]
        x = x / 255.0
        # Add a channels dimension
        x = x[..., tf.newaxis].astype("float32")
        return x, y

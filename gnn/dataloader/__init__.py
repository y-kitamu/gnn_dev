"""dataloader.py

Author : Yusuke Kitamura
Create Date : 2024-02-25 22:47:21
"""

from typing import Tuple

import keras
import tensorflow as tf

from .trainer import DataloaderParams


def get_dataloader(
    train_params: DataloaderParams, test_params
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ """
    return get_mnist(train_params, test_params)


def get_mnist(
    train_params: DataloaderParams, test_params: DataloaderParams
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    train_params.steps_per_epoch = len(x_train) // 32
    test_params.steps_per_epoch = len(x_test) // 32

    return train_ds, test_ds

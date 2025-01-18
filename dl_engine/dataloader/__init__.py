"""dataloader.py

Author : Yusuke Kitamura
Create Date : 2024-02-25 22:47:21
"""

import keras
import tensorflow as tf
from pydantic import BaseModel

from ..base import BaseParams, get_default_params_of
from .base import BaseDataloader


class DataloaderParams(BaseParams):
    pass


class MnistDataloader(BaseDataloader):
    class Params(BaseDataloader.Params):
        pass

    def __init__(self, params: Params, is_train: bool):
        self.params = params
        self.x, self.y = self.get_data(is_train)
        self.iterator = iter(
            tf.data.Dataset.from_tensor_slices((self.x, self.y)).shuffle(len(self.x)).batch(32).repeat()
        )

    @property
    def steps_per_epoch(self):
        return len(self.x) // self.params.batch_size

    def get_next(self):
        x, y = next(self.iterator)
        return {"inputs": x, "y_true": y}

    def get_data(self, is_train: bool):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x = x_train if is_train else x_test
        y = y_train if is_train else y_test
        x = x / 255.0
        # Add a channels dimension
        x = x[..., tf.newaxis].astype("float32")
        return x, y


dataloader_list: dict[str, type[BaseDataloader]] = {"mnist": MnistDataloader}


def get_dataloader(params: DataloaderParams, is_train: bool) -> BaseDataloader:
    """ """
    dataloader_class = dataloader_list[params.name]
    return dataloader_class(dataloader_class.Params(**params.params), is_train)


def get_default_dataloader_params(name: str) -> BaseModel:
    """ """
    return get_default_params_of(name, dataloader_list)

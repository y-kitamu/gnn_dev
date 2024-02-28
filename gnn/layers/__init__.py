"""__init__.py

Author : Yusuke Kitamura
Create Date : 2024-02-28 22:14:35
"""

import keras
from pydantic import BaseModel

from .gcn import GCN
from ..base import BaseParams, get_object_default_params, get_object


class NetworkParams(BaseParams):
    pass


class SimpleModel(keras.Layer):
    class Params(BaseModel):
        pass

    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation="relu")
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128, activation="relu")
        self.d2 = keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


network_list = {"simple": SimpleModel, "gcn": GCN}


def get_model(params: NetworkParams) -> keras.Layer:
    """ """
    return get_object(params, network_list)


def get_model_params(name: str) -> BaseModel:
    """ """
    return get_object_default_params(name, network_list)

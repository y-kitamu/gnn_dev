"""__init__.py

Author : Yusuke Kitamura
Create Date : 2024-02-28 22:14:35
"""

import inspect
from typing import Any, Type

import keras
from pydantic import BaseModel, create_model

from ..base import BaseParams, get_default_params_of, get_object
from .base import BaseNetwork
from .gcn import GCN


class NetworkParams(BaseParams):
    pass


class SimpleModel(BaseNetwork):
    class Params(BaseModel):
        pass

    def __init__(self, params: Params):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation="relu")
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128, activation="relu")
        self.d2 = keras.layers.Dense(10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def get_network_class(network: Type[keras.Layer]):
    """既存のネットワークをラップしたクラスを作成する"""
    param_class_name = network.__code__.co_name + "Params"
    model_param_class = create_model(
        param_class_name,
        **{
            key: (Any, value.default) if value.default is not inspect.Parameter.empty else (Any, ...)
            for key, value in inspect.signature(network).parameters.items()
        }
    )

    class NetworkWrapper(BaseNetwork):

        class Params(model_param_class):
            pass

        def __init__(self, params: Params):
            self.network = network(**params.model_dump())

        def call(self, *args, **kwargs):
            self.network(*args, **kwargs)

    return NetworkWrapper


network_list = {
    "simple": SimpleModel,
    "gcn": GCN,
    "resnet50": get_network_class(keras.applications.ResNet50),
}


def get_model(params: NetworkParams) -> keras.Layer:
    """ """
    return get_object(params, network_list)


def get_default_model_params(name: str) -> BaseModel:
    """ """
    return get_default_params_of(name, network_list)

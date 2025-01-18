"""__init__.py
"""

import inspect
from typing import Any

import keras
from pydantic import BaseModel, create_model

from ..base import BaseNetwork, BaseParams, get_default_params_of, get_object
from .conv_classifier import ConvClassifier
from .gcn import GCN


class NetworkParams(BaseParams):
    pass


def get_network_class(network: type[keras.Layer]):
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
            super().__init__()
            self.network = network(**params.model_dump())

        def call(self, *args, **kwargs) -> dict[str, Any]:
            return {"y_pred": self.network(*args, **kwargs)}

    return NetworkWrapper


network_list = {
    "gcn": GCN,
    "conv_classifier": ConvClassifier,
    "resnet50": get_network_class(keras.applications.ResNet50),
}


def get_model(params: NetworkParams) -> keras.Layer:
    """ """
    return get_object(params, network_list)


def get_default_model_params(name: str) -> BaseModel:
    """ """
    return get_default_params_of(name, network_list)

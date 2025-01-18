"""losses.py
"""

from typing import Any

import keras
import tensorflow as tf
from pydantic import BaseModel

from ..base import BaseLoss, BaseParams, get_default_params_of, get_object
from .crossentropy import BinaryCrossentropyLoss, CategoricalCrossentropyLoss


class LossParams(BaseParams):
    pass


loss_list = {
    "bce": BinaryCrossentropyLoss,
    "cce": CategoricalCrossentropyLoss,
}


def get_loss(params: LossParams) -> BaseLoss:
    """ """
    return get_object(params, loss_list)


def get_default_loss_params(name: str) -> BaseModel:
    """ """
    return get_default_params_of(name, loss_list)

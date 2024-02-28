"""losses.py

Author : Yusuke Kitamura
Create Date : 2024-02-25 22:43:00
"""

import keras
from pydantic import BaseModel

from .trainer import LossParams
from ..base import BaseParams, get_object, get_object_default_params


class LossParams(BaseParams):
    pass


class CrossEntropyLoss(keras.losses.SparseCategoricalCrossentropy):
    class Params(BaseModel):
        pass


loss_list = {"cross_entropy": CrossEntropyLoss}


def get_losses(params: LossParams) -> keras.Loss:
    """ """
    return get_object(params, loss_list)


def get_losses_params(name: str) -> BaseModel:
    """ """
    return get_object_default_params(name, loss_list)

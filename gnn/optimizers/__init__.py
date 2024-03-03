"""optimizers.py

Author : Yusuke Kitamura
Create Date : 2024-02-25 22:44:52
"""

import keras
from pydantic import BaseModel

from ..base import BaseParams, get_object, get_object_default_params


class OptimizerParams(BaseParams):
    pass


class Adam(keras.optimizers.Adam):
    class Params(BaseModel):
        learning_rate: float = 1e-3

    def __init__(self, params: Params):
        super().__init__(params.learning_rate)


optimizer_list = {"adam": Adam}


def get_optimizer(params: OptimizerParams) -> keras.optimizers.Optimizer:
    """ """
    return get_object(params, optimizer_list)


def get_default_optimizer_params(name: str) -> BaseModel:
    return get_object_default_params(name, optimizer_list)

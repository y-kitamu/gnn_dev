"""optimizers.py

Author : Yusuke Kitamura
Create Date : 2024-02-25 22:44:52
"""

import tensorflow as tf
from pydantic import BaseModel

from ..base import BaseParams, get_default_params_of, get_object


class OptimizerParams(BaseParams):
    pass


class Adam(tf.keras.optimizers.Adam):  # kerasではなくtf.kerasを使う
    class Params(BaseModel):
        learning_rate: float = 1e-3

    def __init__(self, params: Params):
        super().__init__(params.learning_rate)


optimizer_list = {"adam": Adam}


def get_optimizer(params: OptimizerParams) -> tf.keras.optimizers.Optimizer:
    """ """
    return get_object(params, optimizer_list)


def get_default_optimizer_params(name: str) -> BaseModel:
    return get_default_params_of(name, optimizer_list)

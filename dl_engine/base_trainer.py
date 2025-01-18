"""base_trainer.py

Author : Yusuke Kitamura
Create Date : 2024-03-03 11:48:08
"""

import keras
import tensorflow as tf
from pydantic import BaseModel

from .dataloader import BaseDataloader
from .layers import BaseNetwork
from .losses import BaseLoss


class BaseTrainer:
    @property
    def params(self) -> BaseModel:
        return self._params

    @property
    def epoch(self) -> tf.Variable:
        return self._epoch

    @property
    def network(self) -> BaseNetwork:
        return self._network

    @property
    def loss(self) -> tf.Tensor:
        return self.output_data["loss"]

    @property
    def loss_fn(self) -> BaseLoss:
        return self._loss_fn

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        return self._optimizer

    @property
    def train_dataloader(self) -> BaseDataloader:
        return self._train_dataloader

    @property
    def test_dataloader(self) -> BaseDataloader:
        return self._test_dataloader

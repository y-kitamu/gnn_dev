"""base_trainer.py
"""

from pathlib import Path

import keras
import tensorflow as tf
from pydantic import BaseModel

from .base_dataloader import BaseDataloader
from .base_layer import BaseNetwork
from .base_loss import BaseLoss


class BaseTrainer:
    @property
    def params(self) -> BaseModel:
        return self._params

    @property
    def pretrain_model_dir(self) -> Path | None:
        return self._params.pretrain_model_dir

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

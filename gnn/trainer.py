"""trainer.py
"""

from pathlib import Path

import tensorflow as tf
import keras
from pydantic import BaseModel


class ModelParams(BaseModel):
    input_shape: tuple
    output_shape: tuple
    model_name: str
    model_params: dict


class DataloaderParams(BaseModel):
    dataset_dir: Path
    batch_size: int = 1
    steps_per_epoch: int = -1


class OptimizerParams(BaseModel):
    optimizer_name: str
    optimizer_params: dict


class LossParams(BaseModel):
    loss_name: str
    loss_params: dict


class TrainParams(BaseModel):
    epochs: int
    learning_rate: float
    model_params: ModelParams
    train_dataloader_params: DataloaderParams
    test_dataloader_params: DataloaderParams
    optimizer_params: OptimizerParams
    loss_params: LossParams


class BaseTrainer:
    def __init__(
        self,
        model: keras.Layer,
        loss: keras.Loss,
        optimizer: keras.Optimizer,
        train_dataloader: tf.data.Dataset,
        test_dataloader: tf.data.Dataset,
        callbacks: keras.callbacks.CallbackList,
        config: TrainParams,
    ):
        """ """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.callbacks = callbacks
        self.callbacks.set_model(self)
        self.config = config

    def train(self):
        """ """
        self.callbacks.on_train_begin()

        for epoch in range(self.config.epochs):
            self.callbacks.on_epoch_begin(epoch)

            assert self.config.train_dataloader_params.steps_per_epoch > 0, "steps_per_epoch must be set"
            assert self.config.test_dataloader_params.steps_per_epoch > 0, "steps_per_epoch must be set"

            # train
            train_iter = iter(self.train_dataloader)
            for step in range(self.config.train_dataloader_params.steps_per_epoch):
                self.callbacks.on_train_batch_begin(step)
                data = next(train_iter)
                self._train_step(data)
                self.callbacks.on_train_batch_end(step)

            # validation
            test_iter = iter(self.test_dataloader)
            for step in range(self.config.test_dataloader_params.steps_per_epoch):
                self.callbacks.on_test_batch_begin(step)
                data = next(test_iter)
                self._test_step(data)
                self.callbacks.on_test_batch_end(step)

            self.callbacks.on_epoch_end(epoch)

        self.callbacks.on_train_end()

    def _train_step(self, data):
        """ """

    def _test_step(self, data):
        """ """

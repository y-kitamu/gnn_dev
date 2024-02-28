"""trainer.py
"""

from pathlib import Path
from typing import List

import keras
import tensorflow as tf
from pydantic import BaseModel

from .optimizers import OptimizerParams
from .losses import LossParams
from .layers import NetworkParams


class DataloaderParams(BaseModel):
    dataset_dir: Path = Path()
    dataset_name: str = ""
    dataset_params: dict = {}
    # batch_size: int = 1
    # steps_per_epoch: int = -1
    # network_input_keys: List[str] = []
    # loss_input_keys: List[str] = []


class TrainParams(BaseModel):
    epochs: int = 5
    network_params: NetworkParams = NetworkParams()
    train_dataloader_params: DataloaderParams = DataloaderParams()
    test_dataloader_params: DataloaderParams = DataloaderParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    loss_params: LossParams = LossParams()


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
        # self.x = [data[key] for key in self.config.train_dataloader_params.network_input_keys]
        # self.y = [data[key] for key in self.config.train_dataloader_params.loss_input_keys]
        self.x, self.y = [data[0]], [data[1]]

        with tf.GradientTape() as tape:
            self.y_pred = self.model(*self.x, training=True)
            print(self.y_pred.shape, self.y[0].shape)
            self.loss_val = self.loss(*(self.y.append(self.y_pred)))

        grads = tape.gradient(self.loss_val, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def _test_step(self, data):
        """ """
        self.x = [data[key] for key in self.config.test_dataloader_params.network_input_keys]
        self.y = [data[key] for key in self.config.test_dataloader_params.loss_input_keys]

        self.y_pred = self.model(*self.x, training=False)
        self.loss_val = self.loss(*(self.y.append(self.y_pred)))

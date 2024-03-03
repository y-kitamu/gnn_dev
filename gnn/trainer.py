"""trainer.py
"""

from typing import List

import tensorflow as tf
from pydantic import BaseModel

from .base_trainer import BaseTrainer
from .callbacks import BaseCallback
from .dataloader import (DataloaderParams, get_dataloader,
                         get_default_dataloader_params)
from .layers import NetworkParams, get_default_model_params, get_model
from .losses import LossParams, get_default_loss_params, get_loss
from .optimizers import (OptimizerParams, get_default_optimizer_params,
                         get_optimizer)


class Trainer(BaseTrainer):
    class Params(BaseModel):
        epochs: int = 5
        network_params: NetworkParams = NetworkParams()
        train_dataloader_params: DataloaderParams = DataloaderParams()
        test_dataloader_params: DataloaderParams = DataloaderParams()
        optimizer_params: OptimizerParams = OptimizerParams()
        loss_params: LossParams = LossParams()
        network_input_keys: List[str] = ["inputs"]
        loss_input_keys: List[str] = ["y_true"]

        @classmethod
        def get_default_params(
            cls,
            network_name: str = "simple",
            optimizer_name: str = "adam",
            loss_name: str = "cross_entropy",
            dataloader_name: str = "mnist",
        ):
            """ """
            return cls(
                network_params=NetworkParams(
                    name=network_name, params=get_default_model_params(network_name).model_dump()
                ),
                optimizer_params=OptimizerParams(
                    name=optimizer_name, params=get_default_optimizer_params(optimizer_name).model_dump()
                ),
                loss_params=LossParams(
                    name=loss_name, params=get_default_loss_params(loss_name).model_dump()
                ),
                train_dataloader_params=DataloaderParams(
                    name=dataloader_name,
                    params=get_default_dataloader_params(dataloader_name).model_dump(),
                ),
                test_dataloader_params=DataloaderParams(
                    name=dataloader_name,
                    params=get_default_dataloader_params(dataloader_name).model_dump(),
                ),
            )

    def __init__(self, params: Params, callbacks: BaseCallback):
        """ """
        self._params = params
        self._network = get_model(params.network_params)
        self._loss = get_loss(params.loss_params)
        self._optimizer = get_optimizer(params.optimizer_params)
        self._train_dataloader = get_dataloader(params.train_dataloader_params, is_train=True)
        self._test_dataloader = get_dataloader(params.test_dataloader_params, is_train=False)
        self._callbacks = callbacks
        self._callbacks.set_trainer(self)

    def train(self):
        """ """
        self._callbacks.on_train_begin()

        for epoch in range(self._params.epochs):
            self._callbacks.on_epoch_begin(epoch)

            assert self.train_dataloader.steps_per_epoch > 0, "steps_per_epoch must be set"
            assert self.test_dataloader.steps_per_epoch > 0, "steps_per_epoch must be set"

            # train
            for step in range(self.train_dataloader.steps_per_epoch):
                self._callbacks.on_train_batch_begin(step)
                data = self.train_dataloader.get_next()
                self._train_step(data)
                self._callbacks.on_train_batch_end(step)

            # validation
            self._callbacks.on_test_begin()
            for step in range(self.test_dataloader.steps_per_epoch):
                self._callbacks.on_test_batch_begin(step)
                data = self.test_dataloader.get_next()
                self._test_step(data)
                self._callbacks.on_test_batch_end(step)
            self._callbacks.on_test_end()

            self._callbacks.on_epoch_end(epoch)

        self._callbacks.on_train_end()

    def _train_step(self, data):
        """ """
        self.x = {key: data[key] for key in self._params.network_input_keys}
        self.y = {key: data[key] for key in self._params.loss_input_keys}

        with tf.GradientTape() as tape:
            self.y_pred = self._network(**self.x, training=True)
            self.y["y_pred"] = self.y_pred
            self.loss_val = self.loss(**self.y)

        grads = tape.gradient(self.loss_val, self._network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._network.trainable_weights))

    def _test_step(self, data):
        """ """
        self.x = {key: data[key] for key in self._params.network_input_keys}
        self.y = {key: data[key] for key in self._params.loss_input_keys}

        self.y_pred = self._network(**self.x, training=False)
        self.y["y_pred"] = self.y_pred
        self.loss_val = self._loss(**self.y)

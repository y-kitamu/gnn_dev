"""trainer.py
"""

from pathlib import Path
from typing import List

import tensorflow as tf
from pydantic import BaseModel, field_serializer, field_validator

from .base_trainer import BaseTrainer
from .callbacks import BaseCallback
from .constants import PROJECT_ROOT
from .dataloader import (DataloaderParams, get_dataloader,
                         get_default_dataloader_params)
from .layers import NetworkParams, get_default_model_params, get_model
from .losses import LossParams, get_default_loss_params, get_loss
from .optimizers import (OptimizerParams, get_default_optimizer_params,
                         get_optimizer)


class Trainer(BaseTrainer):
    class Params(BaseModel):
        epochs: int = 5
        pretrain_model_dir: Path | None = None
        network_params: NetworkParams = NetworkParams()
        train_dataloader_params: DataloaderParams = DataloaderParams()
        test_dataloader_params: DataloaderParams = DataloaderParams()
        optimizer_params: OptimizerParams = OptimizerParams()
        loss_params: LossParams = LossParams()
        network_input_keys: List[str] = ["input"]
        loss_input_keys: List[str] = ["y_true"]
        loss_target_keys: List[str] = ["loss"]

        @field_validator("pretrain_model_dir")
        def _validate_path(cls, v: Path):
            if v is None:
                return v
            if v.is_absolute():
                return v
            return PROJECT_ROOT / v

        @field_serializer("pretrain_model_dir")
        def _serialize_path(self, v: Path, _info):
            if v is None:
                return v
            if v.is_relative_to(PROJECT_ROOT):
                return v.relative_to(PROJECT_ROOT).as_posix()
            return v.as_posix()

        @classmethod
        def get_default_params(
            cls,
            network_name: str = "simple",
            optimizer_name: str = "adam",
            loss_name: str = "bce",
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
        self._loss_fn = get_loss(params.loss_params)
        self._optimizer = get_optimizer(params.optimizer_params)
        self._train_dataloader = get_dataloader(params.train_dataloader_params, is_train=True)
        self._test_dataloader = get_dataloader(params.test_dataloader_params, is_train=False)
        self._callbacks = callbacks
        self._epoch = tf.Variable(0, dtype=tf.int32)

        self._callbacks.set_trainer(self)

    def train(self):
        """ """
        self._callbacks.on_train_begin()

        for epoch in range(self._epoch.numpy(), self._params.epochs):
            self._loss_fn.reset_metrics()
            self._callbacks.on_epoch_begin(epoch)

            assert self.train_dataloader.steps_per_epoch > 0, "steps_per_epoch must be set"
            assert self.test_dataloader.steps_per_epoch > 0, "steps_per_epoch must be set"

            # train
            for step in range(self.train_dataloader.steps_per_epoch):
                self._callbacks.on_train_batch_begin(step)
                self.input_data = self.train_dataloader.get_next()
                self.output_data = self._train_step(self.input_data)
                self._loss_fn.update_metrics(self.output_data["loss_output"])
                self._callbacks.on_train_batch_end(step)

            # validation
            self._callbacks.on_test_begin()
            self._loss_fn.reset_metrics()
            for step in range(self.test_dataloader.steps_per_epoch):
                self._callbacks.on_test_batch_begin(step)
                self.input_data = self.test_dataloader.get_next()
                self.output_data = self._test_step(self.input_data)
                self._loss_fn.update_metrics(self.output_data["loss_output"])
                self._callbacks.on_test_batch_end(step)
            self._callbacks.on_test_end()
            self._loss_fn.reset_metrics()

            self._callbacks.on_epoch_end(epoch)
            self._epoch.assign_add(1)

        self._callbacks.on_train_end()

    @tf.function
    def _train_step(self, data):
        """ """
        with tf.GradientTape() as tape:
            network_input = [data[key] for key in self._params.network_input_keys]
            network_output = self._network(*network_input, training=True)
            network_output = data | network_output
            loss_input = [network_output[key] for key in self._params.loss_input_keys]
            loss_output = self._loss_fn(*loss_input)
            loss = tf.add_n([loss_output[key] for key in self._params.loss_target_keys])

        grads = tape.gradient(loss, self._network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._network.trainable_weights))
        return {
            "loss": loss,
            "network_input": network_input,
            "network_output": network_output,
            "loss_input": loss_input,
            "loss_output": loss_output,
            "grads": grads,
        }

    @tf.function
    def _test_step(self, data):
        """ """
        network_input = [data[key] for key in self._params.network_input_keys]
        network_output = self._network(*network_input, training=True)
        network_output |= data
        loss_input = [network_output[key] for key in self._params.loss_input_keys]
        loss_output = self._loss_fn(*loss_input)
        loss = tf.add_n([loss_output[key] for key in self._params.loss_target_keys])
        return {
            "loss": loss,
            "network_input": network_input,
            "network_output": network_output,
            "loss_input": loss_input,
            "loss_output": loss_output,
        }

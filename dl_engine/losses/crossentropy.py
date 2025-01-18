"""binary_crossentropy.Pu

Author : Yusuke Kitamura
Create Date : 2024-07-15 17:56:38
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""

from typing import Any

import keras
import tensorflow as tf
from pydantic import BaseModel

from .base import BaseLoss


class BaseCrossEntropyLoss(BaseLoss):
    class Params(BaseModel):
        from_logits: bool = True

    def __init__(self, params: Params):
        super().__init__()
        self.params = params
        # self.bce = keras.losses.BinaryCrossentropy(from_logits=params.from_logits)

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def output_keys(self, data) -> list[str]:
        return ["loss"]

    def get_metrics(self) -> dict[str, float]:
        return {name: metric.result() for name, metric in self.metrics.items()}

    def reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset_state()

    def call(self, y_true, y_pred) -> dict[str, Any]:
        loss = self.loss_fn(y_true, y_pred)
        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}


class BinaryCrossentropyLoss(BaseCrossEntropyLoss):
    def __init__(self, params: BaseCrossEntropyLoss.Params):
        super().__init__(params)
        self._loss_fn = keras.losses.BinaryCrossentropy(from_logits=params.from_logits)
        self.metrics = {
            "loss": keras.metrics.Mean(name="loss"),
            "accuracy": keras.metrics.BinaryAccuracy(name="accuracy"),
            "recall": keras.metrics.Recall(name="recall"),
            "precision": keras.metrics.Precision(name="precision"),
        }

    def update_metrics(self, data) -> None:
        self.metrics["loss"](data["loss"])
        y_pred = tf.nn.sigmoid(data["y_pred"])
        self.metrics["accuracy"](data["y_true"], y_pred)
        # y_true_onehot = tf.one_hot(tf.cast(data["y_true"], tf.int32), data["y_pred"].shape[-1])
        self.metrics["recall"](data["y_true"], y_pred)
        self.metrics["precision"](data["y_true"], y_pred)


class CategoricalCrossentropyLoss(BaseCrossEntropyLoss):
    class Params(BaseCrossEntropyLoss.Params):
        num_class: int = 0

    def __init__(self, params: Params):
        super().__init__(params)
        self._loss_fn = keras.losses.CategoricalCrossentropy(from_logits=params.from_logits)
        self.metrics = {
            "loss": keras.metrics.Mean(name="loss"),
            "accuracy": keras.metrics.CategoricalAccuracy(name="accuracy"),
        }
        for i in range(params.num_class):
            self.metrics[f"recall_{i}"] = keras.metrics.Recall(class_id=i)
            self.metrics[f"precision_{i}"] = keras.metrics.Precision(class_id=i)

    def update_metrics(self, data) -> None:
        self.metrics["loss"](data["loss"])
        y_pred = tf.nn.softmax(data["y_pred"])
        self.metrics["accuracy"](data["y_true"], y_pred)
        for i in range(self.params.num_class):
            self.metrics[f"recall_{i}"](data["y_true"], y_pred)
            self.metrics[f"precision_{i}"](data["y_true"], y_pred)

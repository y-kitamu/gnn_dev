"""binary_crossentropy.py

Author : Yusuke Kitamura
Create Date : 2024-07-15 17:56:38
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""

from typing import Any

import keras
import tensorflow as tf
from pydantic import BaseModel

from .base import BaseLoss


class BinaryCrossEntropyLoss(BaseLoss):
    class Params(BaseModel):
        from_logits: bool = False

    def __init__(self, params: Params):
        super().__init__()
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

        self.metrics = {
            "loss": keras.metrics.Mean(name="loss"),
            "accuracy": keras.metrics.BinaryAccuracy(name="accuracy"),
            "recall": keras.metrics.Recall(name="recall"),
            "precision": keras.metrics.Precision(name="precision"),
        }

    @property
    def output_keys(self, data) -> list[str]:
        return ["loss"]

    def update_metrics(self, data) -> None:
        self.metrics["loss"](data["loss"])
        self.metrics["accuracy"](data["y_true"], data["y_pred"])
        y_true_onehot = tf.one_hot(tf.cast(data["y_true"], tf.int32), data["y_pred"].shape[-1])
        self.metrics["recall"](y_true_onehot, data["y_pred"])
        self.metrics["precision"](y_true_onehot, data["y_pred"])

    def get_metrics(self) -> dict[str, float]:
        return {name: metric.result() for name, metric in self.metrics.items()}

    def reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset_state()

    def call(self, y_true, y_pred) -> dict[str, Any]:
        loss = self.bce(y_true, y_pred)
        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

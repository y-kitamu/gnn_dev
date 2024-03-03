"""losses.py

Author : Yusuke Kitamura
Create Date : 2024-02-25 22:43:00
"""

import keras
import tensorflow as tf
from pydantic import BaseModel

from ..base import BaseParams, get_object, get_object_default_params
from .base import BaseLoss


class LossParams(BaseParams):
    pass


class CrossEntropyLoss(keras.losses.SparseCategoricalCrossentropy, BaseLoss):
    class Params(BaseModel):
        from_logits: bool = False

    def __init__(self, params: Params):
        super().__init__(from_logits=params.from_logits)

        self.metrics = {
            "loss": keras.metrics.Mean(name="loss"),
            "accuracy": keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            "recall": keras.metrics.Recall(name="recall"),
            "precision": keras.metrics.Precision(name="precision"),
        }

    def update_metrics(self) -> None:
        self.metrics["loss"](self.loss)
        self.metrics["accuracy"](self.y_true, self.y_pred)
        y_true_onehot = tf.one_hot(tf.cast(self.y_true, tf.int32), self.y_pred.shape[-1])
        self.metrics["recall"](y_true_onehot, self.y_pred)
        self.metrics["precision"](y_true_onehot, self.y_pred)

    def get_metrics(self) -> dict[str, float]:
        return {name: metric.result() for name, metric in self.metrics.items()}

    def reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset_state()

    def call(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.loss = super().call(y_true, y_pred)
        return self.loss


loss_list = {"cross_entropy": CrossEntropyLoss}


def get_loss(params: LossParams) -> BaseLoss:
    """ """
    return get_object(params, loss_list)


def get_default_loss_params(name: str) -> BaseModel:
    """ """
    return get_object_default_params(name, loss_list)

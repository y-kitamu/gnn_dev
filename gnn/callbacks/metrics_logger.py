"""metrics.py

Author : Yusuke Kitamura
Create Date : 2024-02-27 22:24:34
"""

from pathlib import Path

import keras

from ..logger import logger


class MetricsLogger(keras.callbacks.Callback):
    def __init__(self, tensorboard_dir: Path | None = None):
        super().__init__()
        self.tensorboard_dir = tensorboard_dir

        self.train_loss = keras.metrics.Mean(name="train_loss")
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.test_loss = keras.metrics.Mean(name="test_loss")
        self.test_accuracy = keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    def on_epoch_begin(self, epoch: int, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.train_loss.reset_state()
        self.train_accuracy.reset_state()
        self.test_loss.reset_state()
        self.test_accuracy.reset_state()

    def on_train_batch_end(self, batch: int, logs=None):
        super().on_train_batch_end(batch, logs)
        self.train_loss(self.model.loss_val)
        self.train_accuracy(self.model.y_pred, self.model.y)

    def on_test_batch_end(self, batch: int, logs=None):
        super().on_test_batch_end(batch, logs)
        self.test_loss(self.model.loss_val)
        self.test_accuracy(self.model.y_pred, self.model.y)

    def on_epoch_end(self, epoch: int, logs=None):
        super().on_epoch_end(epoch, logs)
        logger.info(
            "Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss : {:.4f}, Test Accuracy : {:.4f}".format(
                epoch,
                self.train_loss.result(),
                self.train_accuracy.result(),
                self.test_loss.result(),
                self.test_accuracy.result(),
            )
        )

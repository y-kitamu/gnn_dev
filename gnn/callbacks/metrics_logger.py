"""metrics.py

Author : Yusuke Kitamura
Create Date : 2024-02-27 22:24:34
"""

from pathlib import Path

from ..logger import logger
from .base import BaseCallback


class MetricsLogger(BaseCallback):
    def __init__(self, tensorboard_dir: Path | None = None):
        super().__init__()
        self.tensorboard_dir = tensorboard_dir
        self.epoch = 0
        self.train_epoch_result = {}
        self.test_epoch_result = {}

    def on_epoch_begin(self, epoch: int, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.epoch = epoch
        self.trainer.network.reset_metrics()
        self.trainer.loss.reset_metrics()

    def on_train_batch_end(self, batch: int, logs=None):
        super().on_train_batch_end(batch, logs)
        self.trainer.network.update_metrics()
        self.trainer.loss.update_metrics()

    def on_test_batch_end(self, batch: int, logs=None):
        super().on_test_batch_end(batch, logs)
        self.trainer.network.update_metrics()
        self.trainer.loss.update_metrics()

    def on_test_begin(self, logs=None):
        # この時点で入っている値はtrainの値
        self.train_epoch_result = self.trainer.network.get_metrics()
        self.train_epoch_result.update(self.trainer.loss.get_metrics())
        log_str = ", ".join([f"{key}: {value:.4f}" for key, value in self.train_epoch_result.items()])
        logger.info(f"Epoch {self.epoch} - train - {log_str}")

        # metrixの値をリセット
        self.trainer.network.reset_metrics()
        self.trainer.loss.reset_metrics()

    def on_epoch_end(self, epoch: int, logs=None):
        super().on_epoch_end(epoch, logs)
        self.test_epoch_result = self.trainer.network.get_metrics()
        self.test_epoch_result.update(self.trainer.loss.get_metrics())
        log_str = ", ".join([f"{key}: {value:.4f}" for key, value in self.test_epoch_result.items()])
        logger.info(f"Epoch {self.epoch} - test - {log_str}")

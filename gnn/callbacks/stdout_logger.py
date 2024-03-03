"""stdout_logger.py

Author : Yusuke Kitamura
Create Date : 2024-03-03 12:23:28
"""

from .base import BaseCallback


class StdoutLogger(BaseCallback):

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch} start")

    def on_train_batch_end(self, batch, logs=None):
        self.train_epoch_result = self.trainer.network.get_metrics()
        self.train_epoch_result.update(self.trainer.loss.get_metrics())
        log_str = ", ".join([f"{key}: {value:.4f}" for key, value in self.train_epoch_result.items()])
        print(
            f"train : {batch:5d} / {self.trainer.train_dataloader.steps_per_epoch:5d}, {log_str}",
            end="\r",
        )

    def on_test_begin(self, logs=None):
        print("test start")

    def on_test_batch_end(self, batch, logs=None):
        self.test_epoch_result = self.trainer.network.get_metrics()
        self.test_epoch_result.update(self.trainer.loss.get_metrics())
        log_str = ", ".join([f"{key}: {value:.4f}" for key, value in self.test_epoch_result.items()])
        print(
            f"test : {batch:5d} / {self.trainer.test_dataloader.steps_per_epoch:5d}, {log_str}",
            end="\r",
        )

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} end")

"""train_logger.py
"""

from pathlib import Path

import tensorflow as tf

from ..logging import logger
from ..base import BaseCallback


class TrainLogger(BaseCallback):
    def __init__(self, tensorboard_dir: Path | None = None):
        super().__init__()
        if tensorboard_dir is not None:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.train_summary_writer = tf.summary.create_file_writer(str(tensorboard_dir / "train"))
            self.test_summary_writer = tf.summary.create_file_writer(str(tensorboard_dir / "test"))
        else:
            self.train_summary_writer = None
            self.test_summary_writer = None
        self.epoch = 0
        self.steps = 0

    def on_train_begin(self, logs=None):
        self.train_steps_per_epoch = self.trainer.train_dataloader.steps_per_epoch
        self.test_steps_per_epoch = self.trainer.test_dataloader.steps_per_epoch
        self.epoch = self.trainer.epoch.numpy()

    # def on_epoch_begin(self, epoch, logs=None):
    #     logger.info(f"Epoch {epoch} start")

    def on_train_batch_end(self, batch, logs=None):
        print(
            "train : {:5d} / {:5d}, {:.3f}".format(batch, self.train_steps_per_epoch, self.trainer.loss),
            end="\r",
        )
        if self.train_summary_writer is not None:
            with self.train_summary_writer.as_default():
                tf.summary.scalar("loss", self.trainer.loss, step=self.steps)
        self.steps += 1

    def on_test_begin(self, logs=None):
        # この時点でget_metricsで取得できる値はtrainの値
        train_epoch_result = self.trainer.loss_fn.get_metrics()
        if self.train_summary_writer is not None:
            with self.train_summary_writer.as_default():
                for name, value in train_epoch_result.items():
                    tf.summary.scalar(name, value, step=self.steps)
        log_str = ", ".join([f"{key}: {value:.4f}" for key, value in train_epoch_result.items()])
        logger.info(f"Epoch {self.epoch} - train - {log_str}")

        # logger.info("test start")

    def on_test_batch_end(self, batch, logs=None):
        print(
            "test : {:5d} / {:5d}, {:.3f}".format(batch, self.test_steps_per_epoch, self.trainer.loss),
            end="\r",
        )

    def on_test_end(self, epoch, logs=None):
        #self.test_epoch_result = self.trainer.network.get_metrics()
        test_epoch_result = self.trainer.loss_fn.get_metrics()
        log_str = ", ".join([f"{key}: {value:.4f}" for key, value in test_epoch_result.items()])
        logger.info(f"Epoch {self.epoch} - test - {log_str}")

        if self.test_summary_writer is not None:
            with self.test_summary_writer.as_default():
                for name, value in test_epoch_result.items():
                    tf.summary.scalar(name, value, step=self.steps)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

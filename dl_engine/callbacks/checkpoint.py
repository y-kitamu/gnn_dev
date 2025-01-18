"""checkpoint.py
"""

from pathlib import Path

import tensorflow as tf

from ..base import BaseCallback, BaseTrainer
from ..logging import logger


class Checkpoint(BaseCallback):

    def __init__(self, output_dir: Path):
        self.pretrain_model_dir = None
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_trainer(self, trainer: BaseTrainer):
        super().set_trainer(trainer)
        self.pretrain_model_dir = self.trainer.pretrain_model_dir
        self.checkpoint = tf.train.Checkpoint(
            model=trainer.network, optimizer=trainer.optimizer, epoch=trainer.epoch
        )
        # restore latest model
        for model_dir in [self.output_dir, self.pretrain_model_dir]:
            if model_dir is None or not model_dir.exists():
                continue
            latest_path = tf.train.latest_checkpoint(model_dir)
            if latest_path is not None:
                logger.info(f"Restore model from : {latest_path}")
                self.checkpoint.restore(latest_path)
                if model_dir == self.pretrain_model_dir:
                    self.trainer.epoch.assign(0)  # reset epoch
                else:
                    self.trainer.epoch.assign_add(1)  # continue epoch
                break

    def on_epoch_end(self, epoch, logs=None):
        save_path = self.checkpoint.save(file_prefix=self.output_dir / "ckpt")
        logger.info(f"Saved model to {save_path}")

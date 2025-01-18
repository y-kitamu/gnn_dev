"""__init__.py

Author : Yusuke Kitamura
Create Date : 2024-02-27 22:24:02
"""

import keras

from .base import BaseCallback
from .checkpoint import Checkpoint
from .train_logger import TrainLogger


class CallbackList(keras.callbacks.CallbackList, BaseCallback):

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

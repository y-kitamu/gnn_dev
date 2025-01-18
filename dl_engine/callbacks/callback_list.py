"""callback_list.py
"""

import keras

from ..base import BaseCallback, BaseTrainer


class CallbackList(keras.callbacks.CallbackList, BaseCallback):

    def set_trainer(self, trainer: BaseTrainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

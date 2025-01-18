"""base.py
"""
import keras

from .base_trainer import BaseTrainer


class BaseCallback(keras.callbacks.Callback):

    def set_trainer(self, trainer: BaseTrainer):
        self.trainer = trainer

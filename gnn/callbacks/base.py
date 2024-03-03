"""base.py

Author : Yusuke Kitamura
Create Date : 2024-03-03 10:41:07
"""

import keras

from ..base_trainer import BaseTrainer


class BaseCallback(keras.callbacks.Callback):

    def set_trainer(self, trainer: BaseTrainer):
        self.trainer = trainer

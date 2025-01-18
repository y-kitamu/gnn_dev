"""base.py

Author : Yusuke Kitamura
Create Date : 2024-03-03 11:17:29
"""

from typing import Any

import keras
from keras.src import ops, tree

from ..base import MetrixMixIn


class BaseLoss(keras.losses.Loss, MetrixMixIn):
    def __call__(self, y_true, y_pred, sample_weight=None):
        with ops.name_scope(self.name):
            y_pred = tree.map_structure(lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred)
            y_true = tree.map_structure(lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_true)

            losses = self.call(y_true, y_pred)
            return losses

    def call(self, *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def output_keys(self) -> list[str]:
        raise NotImplementedError

"""base.py
"""

from typing import Any

import keras
from keras.src import ops, tree


class MetrixMixIn:
    def update_metrics(self, data: dict[str, Any]) -> None:
        pass

    def get_metrics(self) -> dict[str, float]:
        return dict()

    def reset_metrics(self) -> None:
        pass


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

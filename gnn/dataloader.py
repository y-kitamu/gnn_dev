"""dataloader.py

Author : Yusuke Kitamura
Create Date : 2024-02-25 22:47:21
"""

from typing import Tuple

import keras
import tensorflow as tf

from .trainer import DataloaderParams


def get_dataloader(params: DataloaderParams) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ """
    pass

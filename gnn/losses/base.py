"""base.py

Author : Yusuke Kitamura
Create Date : 2024-03-03 11:17:29
"""

import keras

from ..base import MetrixMixIn


class BaseLoss(keras.losses.Loss, MetrixMixIn):
    pass

"""base.py

Author : Yusuke Kitamura
Create Date : 2024-02-28 22:21:41
"""

import keras

from ..base import MetrixMixIn


class BaseNetwork(keras.Layer, MetrixMixIn):
    pass

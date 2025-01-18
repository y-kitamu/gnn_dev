"""convnorm.py

Author : Yusuke Kitamura
Create Date : 2024-07-25 21:52:10
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""

import keras


class ConvNorm(keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        super(ConvNorm, self).__init__(**kwargs)
        self.conv = keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.norm = keras.layers.BatchNormalization()
        if activation is not None:
            self.activation = keras.layers.Activation(activation)

    def call(self, x):
        x = self.norm(self.conv(x))
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x

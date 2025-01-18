"""conv_classifier.py
"""

import tensorflow as tf
from pydantic import BaseModel

from ..base import BaseNetwork
from .base_layers import ConvBlock


class ConvClassifier(BaseNetwork):
    class Params(BaseModel):
        pool_num: int = 2
        conv_per_pool: int = 2
        base_filters: int = 8
        num_classes: int = 10

    def __init__(self, params: Params):
        super().__init__()
        self.params = params
        self.conv_blocks = [
            [ConvBlock(params.base_filters * (2**i)) for _ in range(params.conv_per_pool)]
            for i in range(params.pool_num + 1)
        ]
        self.classifier = tf.keras.layers.Dense(params.num_classes)

    @property
    def output_keys(self) -> list[str]:
        return ["y_pred"]

    def call(self, inputs: tf.Tensor) -> dict[str, tf.Tensor]:
        x = inputs
        for idx, conv_block in enumerate(self.conv_blocks):
            for conv in conv_block:
                x = conv(x)
            if idx < len(self.conv_blocks) - 1:
                x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.classifier(x)
        return {"y_pred": x}

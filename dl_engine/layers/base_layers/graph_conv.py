"""gcn.py

Author : Yusuke Kitamura
Create Date : 2024-02-28 22:15:26
"""

import keras
import numpy as np
import tensorflow as tf

from ...logging import logger


class GraphConv(keras.layers.Layer):
    """Graph convolution layer."""

    def __init__(
        self,
        out_dim: int,
        num_nodes: int,
        adjacency_list: np.ndarray,
        bias: bool = True,
        activation: keras.layers.Activation | None = None,
        **kwargs
    ):
        """
        Args:
            out_dim: output dimension
            adjacency_list (np.ndarray): list of adjacency nodes (num_edges, 2).
                 adjacency_list[i, 0] and adjacency_list[i, 1] are the source and destination nodes of the i-th edge.
        """
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.adjacency_list = adjacency_list
        if not np.issubdtype(self.adjacency_list.dtype, np.integer):
            logger.warning("adjacency_list is not integer type. It will be converted to integer type.")
            self.adjacency_list = self.adjacency_list.astype(int)

        # Compute edge weights.
        num_adjacent_nodes = np.bincount(self.adjacency_list[:, 0], minlength=num_nodes)
        node_weights = np.sqrt(1 / (num_adjacent_nodes + 1))
        self.edge_weights = (
            node_weights[self.adjacency_list[:, 0]] * node_weights[self.adjacency_list[:, 1]]
        ).reshape(1, -1, 1)
        self.node_weights = tf.convert_to_tensor((node_weights**2).reshape(1, -1, 1), dtype=tf.float32)

        self.linear = keras.layers.Dense(out_dim, use_bias=False)
        if bias:
            self.bias = tf.Variable(tf.zeros((1, 1, out_dim), dtype=tf.float32), trainable=True)
        self.activation = activation

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Args:
            inputs: (batch_size, num_nodes, in_dim)
        """
        x = self.linear(inputs, *args, **kwargs)
        if self.activation is not None:
            x = self.activation(x)
        gathered = tf.gather(x, self.adjacency_list[:, 0], axis=1) * self.edge_weights
        gathered = tf.transpose(gathered, perm=[1, 0, 2])
        gathered = tf.math.unsorted_segment_sum(gathered, self.adjacency_list[:, 1], inputs.shape[1])
        x = self.node_weights * x + tf.transpose(gathered, perm=[1, 0, 2])
        if hasattr(self, "bias"):
            x += self.bias

        return x

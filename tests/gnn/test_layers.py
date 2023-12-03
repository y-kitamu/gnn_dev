"""test_layers.py

Author : Yusuke Kitamura
Create Date : 2023-12-03 09:13:56
"""
from gnn import layers

import numpy as np


def test_graph_conv():
    layer = layers.GraphConv(out_dim=4, num_nodes=4, adjacency_list=np.array([[0, 1], [1, 0], [1, 2]]))

    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    output_tensor = layer(input_tensor, training=False).numpy()

    assert output_tensor.shape == (1, 4, 4)
    assert layer.edge_weights.shape == (1, 3, 1)
    assert abs(layer.node_weights[0, 0, 0] - 1 / 2) < 1e-5
    assert abs(layer.node_weights[0, 1, 0] - 1 / 3) < 1e-5
    assert abs(layer.node_weights[0, 2, 0] - 1) < 1e-5
    assert abs(layer.edge_weights[0, 0, 0] - 1 / (np.sqrt(2) * np.sqrt(3))) < 1e-5
    assert abs(layer.edge_weights[0, 1, 0] - 1 / (np.sqrt(3) * np.sqrt(2))) < 1e-5
    assert abs(layer.edge_weights[0, 2, 0] - 1 / (np.sqrt(3) * 1)) < 1e-5

    layer.linear.kernel.assign(np.ones((3, 4)))
    layer.bias.assign(np.ones((1, 1, 4)))
    output_tensor = layer(input_tensor, training=False).numpy()
    assert (
        abs(output_tensor[0, 0, 0] - (6.0 / 2.0 + 15.0 / (np.sqrt(2) * np.sqrt(3))) - 1.0) < 1e-5
    ), output_tensor[0, 0, 0]
    assert abs(output_tensor[0, 0, 1] - output_tensor[0, 0, 0]) < 1e-5
    assert abs(output_tensor[0, 0, 2] - output_tensor[0, 0, 0]) < 1e-5
    assert abs(output_tensor[0, 0, 3] - output_tensor[0, 0, 0]) < 1e-5

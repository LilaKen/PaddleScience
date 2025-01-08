from typing import Optional

import pytest
import paddle

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.explain import Explanation
from paddle_geometric.explain.config import MaskType
from paddle_geometric.nn import SAGEConv, to_hetero
from paddle_geometric.testing import get_random_edge_index


@pytest.fixture()
def data():
    return Data(
        x=paddle.randn(shape=[4, 3]),
        edge_index=get_random_edge_index(4, 4, num_edges=6),
        edge_attr=paddle.randn(shape=[6, 3]),
    )


@pytest.fixture()
def hetero_data():
    data = HeteroData()
    data['paper'].x = paddle.randn(shape=[8, 16])
    data['author'].x = paddle.randn(shape=[10, 8])

    data['paper', 'paper'].edge_index = get_random_edge_index(8, 8, 10)
    data['paper', 'paper'].edge_attr = paddle.randn(shape=[10, 16])
    data['paper', 'author'].edge_index = get_random_edge_index(8, 10, 10)
    data['paper', 'author'].edge_attr = paddle.randn(shape=[10, 8])
    data['author', 'paper'].edge_index = get_random_edge_index(10, 8, 10)
    data['author', 'paper'].edge_attr = paddle.randn(shape=[10, 8])

    return data


@pytest.fixture()
def hetero_model():
    return HeteroSAGE


class GraphSAGE(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(paddle.nn.Layer):
    def __init__(self, metadata):
        super().__init__()
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)
        self.lin = paddle.nn.Linear(32, 1)

    def forward(self, x_dict, edge_index_dict,
                additonal_arg=None) -> paddle.to_tensor:
        return self.lin(self.graph_sage(x_dict, edge_index_dict)['paper'])


@pytest.fixture()
def check_explanation():
    def _check_explanation(
        explanation: Explanation,
        node_mask_type: Optional[MaskType],
        edge_mask_type: Optional[MaskType],
    ):
        if node_mask_type == MaskType.attributes:
            assert explanation.node_mask.shape== explanation.x.size()
            assert explanation.node_mask.min() >= 0
            assert explanation.node_mask.max() <= 1
        elif node_mask_type == MaskType.object:
            assert explanation.node_mask.shape== (explanation.num_nodes, 1)
            assert explanation.node_mask.min() >= 0
            assert explanation.node_mask.max() <= 1
        elif node_mask_type == MaskType.common_attributes:
            assert explanation.node_mask.shape== (1, explanation.x.shape[-1])
            assert explanation.node_mask.min() >= 0
            assert explanation.node_mask.max() <= 1
        elif node_mask_type is None:
            assert 'node_mask' not in explanation

        if edge_mask_type == MaskType.object:
            assert explanation.edge_mask.shape== (explanation.num_edges, )
            assert explanation.edge_mask.min() >= 0
            assert explanation.edge_mask.max() <= 1
        elif edge_mask_type is None:
            assert 'edge_mask' not in explanation

    return _check_explanation

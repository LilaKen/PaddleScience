import pytest
import paddle

from paddle_geometric.data import Data
from paddle_geometric.utils import train_test_split_edges


def test_train_test_split_edges():
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    edge_attr = paddle.arange(edge_index.shape[1])
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = edge_index.max().item() + 1

    with pytest.warns(UserWarning, match='deprecated'):
        data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.3)

    assert len(data) == 10
    assert data.val_pos_edge_index.shape == [2, 2]
    assert data.val_neg_edge_index.shape == [2, 2]
    assert data.test_pos_edge_index.shape == [2, 3]
    assert data.test_neg_edge_index.shape == [2, 3]
    assert data.train_pos_edge_index.shape == [2, 10]
    assert data.train_neg_adj_mask.shape == [11, 11]
    assert paddle.sum(data.train_neg_adj_mask).item() == (11 ** 2 - 11) / 2 - 4 - 6 - 5
    assert data.train_pos_edge_attr.shape == [10]
    assert data.val_pos_edge_attr.shape == [2]
    assert data.test_pos_edge_attr.shape == [3]


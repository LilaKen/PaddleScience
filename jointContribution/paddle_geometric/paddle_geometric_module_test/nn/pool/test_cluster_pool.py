import pytest
import paddle

from paddle_geometric.nn import ClusterPooling
from paddle_geometric.testing import withPackage


@withPackage('scipy')
@pytest.mark.parametrize('edge_score_method', [
    'tanh',
    'sigmoid',
    'log_softmax',
])
def test_cluster_pooling(edge_score_method):
    x = paddle.to_tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [-1.0]])
    edge_index = paddle.to_tensor([
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6],
        [1, 2, 3, 6, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 0],
    ])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 0])

    op = ClusterPooling(in_channels=1, edge_score_method=edge_score_method)
    assert str(op) == 'ClusterPooling(1)'
    op.reset_parameters()

    x, edge_index, batch, unpool_info = op(x, edge_index, batch)
    assert x.shape[0] <= 7
    assert edge_index.shape[0] == 2
    if edge_index.numel() > 0:
        assert edge_index.min() >= 0
        assert edge_index.max() < x.shape[0]
    assert batch.shape== (x.shape[0], )

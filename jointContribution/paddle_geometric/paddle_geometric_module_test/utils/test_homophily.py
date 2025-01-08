import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import homophily


def test_homophily():
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 2, 0, 4]])
    y = paddle.to_tensor([0, 0, 0, 0, 1])
    batch = paddle.to_tensor([0, 0, 0, 1, 1])
    row, col = edge_index
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor(row=row, col=col, sparse_sizes=(5, 5))

    method = 'edge'
    assert pytest.approx(homophily(edge_index, y, method=method)) == 0.75
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert pytest.approx(homophily(adj, y, method=method)) == 0.75
    assert homophily(edge_index, y, batch, method).tolist() == [1., 0.]

    method = 'node'
    assert pytest.approx(homophily(edge_index, y, method=method)) == 0.6
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert pytest.approx(homophily(adj, y, method=method)) == 0.6
    assert homophily(edge_index, y, batch, method).tolist() == [1., 0.]

    method = 'edge_insensitive'
    assert pytest.approx(homophily(edge_index, y, method=method)) == 0.1999999
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert pytest.approx(homophily(adj, y, method=method)) == 0.1999999
    assert homophily(edge_index, y, batch, method).tolist() == [0., 0.]

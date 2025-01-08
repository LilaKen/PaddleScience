import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import assortativity


def test_assortativity():
    # Completely assortative graph:
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    out = assortativity(edge_index)
    assert pytest.approx(out, abs=1e-5) == 1.0

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=[6, 6])
        out = assortativity(adj)
        assert pytest.approx(out, abs=1e-5) == 1.0

    # Completely disassortative graph:
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 5, 5, 5, 5],
                               [5, 5, 5, 5, 5, 0, 1, 2, 3, 4]])
    out = assortativity(edge_index)
    assert pytest.approx(out, abs=1e-5) == -1.0

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=[6, 6])
        out = assortativity(adj)
        assert pytest.approx(out, abs=1e-5) == -1.0

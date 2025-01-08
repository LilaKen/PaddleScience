import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import LINKX
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor


@pytest.mark.parametrize('num_edge_layers', [1, 2])
def test_linkx(num_edge_layers):
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2], [1, 2, 3]])
    edge_weight = paddle.rand(edge_index.shape[1])

    model = LINKX(num_nodes=4, in_channels=16, hidden_channels=32,
                  out_channels=8, num_layers=2,
                  num_edge_layers=num_edge_layers)
    assert str(model) == 'LINKX(num_nodes=4, in_channels=16, out_channels=8)'

    out = model(x, edge_index)
    assert out.shape== (4, 8)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert paddle.allclose(out, model(x, adj.t()), atol=1e-6)

    if is_full_test():
        jit = paddle.jit.to_static(model)
        assert paddle.allclose(jit(x, edge_index), out)

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj.t()), out, atol=1e-6)

    out = model(None, edge_index)
    assert out.shape== (4, 8)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(out, model(None, adj.t()), atol=1e-6)

    out = model(x, edge_index, edge_weight)
    assert out.shape== (4, 8)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(4, 4))
        assert paddle.allclose(model(x, adj.t()), out, atol=1e-6)

    out = model(None, edge_index, edge_weight)
    assert out.shape== (4, 8)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(model(None, adj.t()), out, atol=1e-6)

import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import NeuralFingerprint
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor


@pytest.mark.parametrize('batch', [None, paddle.to_tensor([0, 1, 1])])
def test_neural_fingerprint(batch):
    x = paddle.randn(shape=[3, 7])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = NeuralFingerprint(7, 16, out_channels=5, num_layers=4)
    assert str(model) == 'NeuralFingerprint(7, 5, num_layers=4)'
    model.reset_parameters()

    out = model(x, edge_index, batch)
    assert out.shape== (1, 5) if batch is None else (2, 5)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))
        assert paddle.allclose(model(x, adj.t(), batch), out)

    if is_full_test():
        jit = paddle.jit.export(model)
        assert paddle.allclose(jit(x, edge_index, batch), out)

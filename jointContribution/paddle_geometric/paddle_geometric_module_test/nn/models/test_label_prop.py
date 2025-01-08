import paddle

import paddle_geometric.typing
from paddle_geometric.nn.models import LabelPropagation
from paddle_geometric.typing import SparseTensor


def test_label_prop():
    y = paddle.to_tensor([1, 0, 0, 2, 1, 1])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = paddle.randint(0, 2, (6, ), dtype=paddle.bool)

    model = LabelPropagation(num_layers=2, alpha=0.5)
    assert str(model) == 'LabelPropagation(num_layers=2, alpha=0.5)'

    # Test without mask:
    out = model(y, edge_index)
    assert out.shape== (6, 3)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert paddle.allclose(model(y, adj.t()), out)

    # Test with mask:
    out = model(y, edge_index, mask)
    assert out.shape== (6, 3)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(model(y, adj.t(), mask), out)

    # Test post step:
    out = model(y, edge_index, mask, post_step=lambda y: paddle.zeros_like(y))
    assert paddle.sum(out) == 0.

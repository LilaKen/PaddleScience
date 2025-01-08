import paddle

import paddle_geometric.typing
from paddle_geometric.nn import RECT_L
from paddle_geometric.testing import is_full_test
from paddle_geometric.typing import SparseTensor


def test_rect():
    x = paddle.randn(shape=[6, 8])
    y = paddle.to_tensor([1, 0, 0, 2, 1, 1])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = paddle.randint(0, 2, (6, ), dtype=paddle.bool)

    model = RECT_L(8, 16)
    assert str(model) == 'RECT_L(8, 16)'

    out = model(x, edge_index)
    assert out.shape== (6, 8)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert paddle.allclose(out, model(x, adj.t()), atol=1e-6)

    # Test `embed`:
    embed_out = model.embed(x, edge_index)
    assert embed_out.shape== (6, 16)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(embed_out, model.embed(x, adj.t()), atol=1e-6)

    # Test `get_semantic_labels`:
    labels_out = model.get_semantic_labels(x, y, mask)
    assert labels_out.shape== (int(mask.sum()), 8)

    if is_full_test():
        jit = paddle.jit.to_static(model)
        assert paddle.allclose(jit(x, edge_index), out, atol=1e-6)
        assert paddle.allclose(embed_out, jit.embed(x, edge_index), atol=1e-6)
        assert paddle.allclose(labels_out, jit.get_semantic_labels(x, y, mask))

        if paddle_geometric.typing.WITH_PADDLE_SPARSE:
            assert paddle.allclose(jit(x, adj.t()), out, atol=1e-6)
            assert paddle.allclose(embed_out, jit.embed(x, adj.t()), atol=1e-6)
            assert paddle.allclose(labels_out,
                                  jit.get_semantic_labels(x, y, mask))

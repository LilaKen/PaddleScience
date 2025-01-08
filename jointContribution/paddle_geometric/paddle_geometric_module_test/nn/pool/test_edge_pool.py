import paddle

from paddle_geometric.nn import EdgePooling
from paddle_geometric.testing import is_full_test
from paddle_geometric.utils import scatter


def test_compute_edge_score_softmax():
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = paddle.randn(shape=[edge_index.shape[1]])
    e = EdgePooling.compute_edge_score_softmax(raw, edge_index, 6)
    assert paddle.all(e >= 0) and paddle.all(e <= 1)

    # Test whether all incoming edge scores sum up to one.
    assert paddle.allclose(
        scatter(e, edge_index[1], reduce='sum'),
        paddle.ones(6),
    )

    if is_full_test():
        jit = paddle.jit.to_static(EdgePooling.compute_edge_score_softmax)
        assert paddle.allclose(jit(raw, edge_index, 6), e)


def test_compute_edge_score_tanh():
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = paddle.randn(shape=[edge_index.shape[1]])
    e = EdgePooling.compute_edge_score_tanh(raw, edge_index, 6)
    assert paddle.all(e >= -1) and paddle.all(e <= 1)
    assert paddle.all(paddle.argsort(raw) == paddle.argsort(e))

    if is_full_test():
        jit = paddle.jit.to_static(EdgePooling.compute_edge_score_tanh)
        assert paddle.allclose(jit(raw, edge_index, 6), e)


def test_compute_edge_score_sigmoid():
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = paddle.randn(shape=[edge_index.shape[1]])
    e = EdgePooling.compute_edge_score_sigmoid(raw, edge_index, 6)
    assert paddle.all(e >= 0) and paddle.all(e <= 1)
    assert paddle.all(paddle.argsort(raw) == paddle.argsort(e))

    if is_full_test():
        jit = paddle.jit.to_static(EdgePooling.compute_edge_score_sigmoid)
        assert paddle.allclose(jit(raw, edge_index, 6), e)


def test_edge_pooling():
    x = paddle.to_tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [-1.0]])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 0]])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 0])

    op = EdgePooling(in_channels=1)
    assert str(op) == 'EdgePooling(1)'

    # Setting parameters fixed so we can test the expected outcome:
    op.lin.weight.data.fill_(1.)
    op.lin.bias.data.fill_(0.)

    # Test pooling:
    new_x, new_edge_index, new_batch, unpool_info = op(x, edge_index, batch)

    assert new_x.shape[0] == new_batch.shape[0] == 4
    assert new_edge_index.tolist() == [[0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 2]]
    assert new_batch.tolist() == [1, 0, 0, 0]

    if is_full_test():
        jit = paddle.jit.to_static(op)
        out = jit(x, edge_index, batch)
        assert paddle.allclose(new_x, out[0])
        assert paddle.equal(new_edge_index, out[1])
        assert paddle.equal(new_batch, out[2])

    # Test unpooling:
    out = op.unpool(new_x, unpool_info)
    assert out[0].shape== x.size()
    assert out[0].tolist() == [[1], [1], [5], [5], [9], [9], [-1]]
    assert paddle.equal(out[1], edge_index)
    assert paddle.equal(out[2], batch)

    if is_full_test():
        jit = paddle.jit.export(op)
        out = jit.unpool(new_x, unpool_info)
        assert out[0].shape== x.size()
        assert out[0].tolist() == [[1], [1], [5], [5], [9], [9], [-1]]
        assert paddle.equal(out[1], edge_index)
        assert paddle.equal(out[2], batch)

    # Test edge cases.
    x = paddle.to_tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1])
    new_x, new_edge_index, new_batch, _ = op(x, edge_index, batch)

    assert new_x.shape[0] == new_batch.shape[0] == 3
    assert new_batch.tolist() == [1, 0, 0]
    assert new_edge_index.tolist() == [[0, 1, 1, 2, 2], [0, 1, 2, 1, 2]]

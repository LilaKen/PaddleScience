import paddle

from paddle_geometric.nn.aggr import Set2Set


def test_set2set():
    set2set = Set2Set(in_channels=2, processing_steps=1)
    assert str(set2set) == 'Set2Set(2, 4)'

    N = 4
    x_1, batch_1 = paddle.randn(shape=[N, 2]), paddle.zeros(N, dtype=paddle.int64)
    out_1 = set2set(x_1, batch_1).view(-1)

    N = 6
    x_2, batch_2 = paddle.randn(shape=[N, 2]), paddle.zeros(N, dtype=paddle.int64)
    out_2 = set2set(x_2, batch_2).view(-1)

    x, batch = paddle.concat([x_1, x_2]), paddle.concat([batch_1, batch_2 + 1])
    out = set2set(x, batch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out_1, out[0])
    assert paddle.allclose(out_2, out[1])

    x, batch = paddle.concat([x_2, x_1]), paddle.concat([batch_2, batch_1 + 1])
    out = set2set(x, batch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out_1, out[1])
    assert paddle.allclose(out_2, out[0])

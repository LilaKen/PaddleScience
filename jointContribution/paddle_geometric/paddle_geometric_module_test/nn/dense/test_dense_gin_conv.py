import paddle
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

from paddle_geometric.nn import DenseGINConv, GINConv
from paddle_geometric.testing import is_full_test


def test_dense_gin_conv():
    channels = 16
    nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
    sparse_conv = GINConv(nn)
    dense_conv = DenseGINConv(nn)
    dense_conv = DenseGINConv(nn, train_eps=True)
    assert str(dense_conv) == (
        'DenseGINConv(nn=Sequential(\n'
        '  (0): Linear(in_features=16, out_features=16, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=16, out_features=16, bias=True)\n'
        '))')

    x = paddle.randn(shape=[(5, channels]))
    edge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 2, 3, 4],
                               [1, 2, 0, 2, 0, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)
    assert sparse_out.shape== (5, channels)

    x = paddle.concat([x, x.new_zeros(1, channels)], dim=0).view(2, 3, channels)
    adj = paddle.to_tensor([
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ])
    mask = paddle.to_tensor([[1, 1, 1], [1, 1, 0]], dtype=paddle.bool)

    dense_out = dense_conv(x, adj, mask)
    assert dense_out.shape== (2, 3, channels)

    if is_full_test():
        jit = paddle.jit.to_static(dense_conv)
        assert paddle.allclose(jit(x, adj, mask), dense_out)

    assert dense_out[1, 2].abs().sum().item() == 0
    dense_out = dense_out.view(6, channels)[:-1]
    assert paddle.allclose(sparse_out, dense_out, atol=1e-04)


def test_dense_gin_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
    conv = DenseGINConv(nn)

    x = paddle.randn(shape=[batch_size, num_nodes, channels])
    adj = paddle.to_tensor([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    assert conv(x, adj).shape== (batch_size, num_nodes, channels)
    mask = paddle.to_tensor([1, 1, 1], dtype=paddle.bool)
    assert conv(x, adj, mask).shape== (batch_size, num_nodes, channels)

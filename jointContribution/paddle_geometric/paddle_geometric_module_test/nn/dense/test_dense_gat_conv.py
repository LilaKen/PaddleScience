import pytest
import paddle

from paddle_geometric.nn import DenseGATConv, GATConv
from paddle_geometric.testing import is_full_test


@pytest.mark.parametrize('heads', [1, 4])
@pytest.mark.parametrize('concat', [True, False])
def test_dense_gat_conv(heads, concat):
    channels = 16
    sparse_conv = GATConv(channels, channels, heads=heads, concat=concat)
    dense_conv = DenseGATConv(channels, channels, heads=heads, concat=concat)
    assert str(dense_conv) == f'DenseGATConv(16, 16, heads={heads})'

    # Ensure same weights and bias:
    dense_conv.lin = sparse_conv.lin
    dense_conv.att_src = sparse_conv.att_src
    dense_conv.att_dst = sparse_conv.att_dst
    dense_conv.bias = sparse_conv.bias

    x = paddle.randn(shape=[(5, channels]))
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])

    sparse_out = sparse_conv(x, edge_index)

    x = paddle.concat([x, x.new_zeros(1, channels)], dim=0).view(2, 3, channels)
    adj = paddle.to_tensor([
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    ])
    mask = paddle.to_tensor([[1, 1, 1], [1, 1, 0]], dtype=paddle.bool)

    dense_out = dense_conv(x, adj, mask)

    if is_full_test():
        jit = paddle.jit.to_static(dense_conv)
        assert paddle.allclose(jit(x, adj, mask), dense_out)

    assert dense_out[1, 2].abs().sum() == 0
    dense_out = dense_out.view(6, dense_out.shape[-1])[:-1]
    assert paddle.allclose(sparse_out, dense_out, atol=1e-4)


def test_dense_gat_conv_with_broadcasting():
    batch_size, num_nodes, channels = 8, 3, 16
    conv = DenseGATConv(channels, channels, heads=4)

    x = paddle.randn(shape=[batch_size, num_nodes, channels])
    adj = paddle.to_tensor([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    assert conv(x, adj).shape== (batch_size, num_nodes, 64)
    mask = paddle.to_tensor([1, 1, 1], dtype=paddle.bool)
    assert conv(x, adj, mask).shape== (batch_size, num_nodes, 64)

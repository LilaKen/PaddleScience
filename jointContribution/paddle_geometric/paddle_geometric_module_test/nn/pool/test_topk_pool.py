import paddle

import paddle_geometric.typing
from paddle_geometric.nn.pool import TopKPooling
from paddle_geometric.nn.pool.connect.filter_edges import filter_adj
from paddle_geometric.testing import is_full_test


def test_filter_adj():
    edge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                               [1, 3, 0, 2, 1, 3, 0, 2]])
    edge_attr = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    perm = paddle.to_tensor([2, 3])

    out = filter_adj(edge_index, edge_attr, perm)
    assert out[0].tolist() == [[0, 1], [1, 0]]
    assert out[1].tolist() == [6.0, 8.0]

    if is_full_test():
        jit = paddle.jit.to_static(filter_adj)

        out = jit(edge_index, edge_attr, perm)
        assert out[0].tolist() == [[0, 1], [1, 0]]
        assert out[1].tolist() == [6.0, 8.0]


def test_topk_pooling():
    in_channels = 16
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = paddle.randn(shape=[(num_nodes, in_channels]))

    pool1 = TopKPooling(in_channels, ratio=0.5)
    assert str(pool1) == 'TopKPooling(16, ratio=0.5, multiplier=1.0)'
    out1 = pool1(x, edge_index)
    assert out1[0].shape== (num_nodes // 2, in_channels)
    assert out1[1].shape== (2, 2)

    pool2 = TopKPooling(in_channels, ratio=None, min_score=0.1)
    assert str(pool2) == 'TopKPooling(16, min_score=0.1, multiplier=1.0)'
    out2 = pool2(x, edge_index)
    assert out2[0].shape[0] <= x.shape[0] and out2[0].shape[1] == (16)
    assert out2[1].shape[0] == 2 and out2[1].shape[1] <= edge_index.shape[1]

    pool3 = TopKPooling(in_channels, ratio=2)
    assert str(pool3) == 'TopKPooling(16, ratio=2, multiplier=1.0)'
    out3 = pool3(x, edge_index)
    assert out3[0].shape== (2, in_channels)
    assert out3[1].shape== (2, 2)

    if is_full_test():
        jit1 = paddle.jit.to_static(pool1)
        assert paddle.allclose(jit1(x, edge_index)[0], out1[0])

        jit2 = paddle.jit.to_static(pool2)
        assert paddle.allclose(jit2(x, edge_index)[0], out2[0])

        jit3 = paddle.jit.to_static(pool3)
        assert paddle.allclose(jit3(x, edge_index)[0], out3[0])

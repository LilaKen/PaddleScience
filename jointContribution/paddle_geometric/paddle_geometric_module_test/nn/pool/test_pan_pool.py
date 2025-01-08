import paddle

import paddle_geometric.typing
from paddle_geometric.nn import PANConv, PANPooling
from paddle_geometric.testing import is_full_test, withPackage


@withPackage('torch_sparse')
def test_pan_pooling():
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = paddle.randn(shape=[num_nodes, 16])

    conv = PANConv(16, 32, filter_size=2)
    pool = PANPooling(32, ratio=0.5)
    assert str(pool) == 'PANPooling(32, ratio=0.5, multiplier=1.0)'

    x, M = conv(x, edge_index)
    h, edge_index, edge_weight, batch, perm, score = pool(x, M)

    assert h.shape== (2, 32)
    assert edge_index.shape== (2, 4)
    assert edge_weight.shape== (4, )
    assert perm.shape== (2, )
    assert score.shape== (2, )

    if is_full_test():
        jit = paddle.jit.to_static(pool)
        out = jit(x, M)
        assert paddle.allclose(h, out[0])
        assert paddle.equal(edge_index, out[1])
        assert paddle.allclose(edge_weight, out[2])
        assert paddle.equal(batch, out[3])
        assert paddle.equal(perm, out[4])
        assert paddle.allclose(score, out[5])

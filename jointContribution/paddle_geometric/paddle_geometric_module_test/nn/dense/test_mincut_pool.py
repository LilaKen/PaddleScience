import math

import paddle

from paddle_geometric.nn import dense_mincut_pool
from paddle_geometric.testing import is_full_test


def test_dense_mincut_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = paddle.randn(shape=[(batch_size, num_nodes, channels]))
    adj = paddle.ones((batch_size, num_nodes, num_nodes))
    s = paddle.randn(shape=[(batch_size, num_nodes, num_clusters]))
    mask = paddle.randint(0, 2, (batch_size, num_nodes), dtype=paddle.bool)

    x_out, adj_out, mincut_loss, ortho_loss = dense_mincut_pool(
        x, adj, s, mask)
    assert x_out.shape== (2, 10, 16)
    assert adj_out.shape== (2, 10, 10)
    assert -1 <= mincut_loss <= 0
    assert 0 <= ortho_loss <= 2

    if is_full_test():
        jit = paddle.jit.to_static(dense_mincut_pool)

        x_jit, adj_jit, mincut_loss, ortho_loss = jit(x, adj, s, mask)
        assert x_jit.shape== (2, 10, 16)
        assert adj_jit.shape== (2, 10, 10)
        assert -1 <= mincut_loss <= 0
        assert 0 <= ortho_loss <= math.sqrt(2)

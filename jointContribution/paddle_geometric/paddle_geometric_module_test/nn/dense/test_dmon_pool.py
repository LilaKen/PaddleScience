import math

import paddle

from paddle_geometric.nn import DMoNPooling


def test_dmon_pooling():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = paddle.randn(shape=[(batch_size, num_nodes, channels]))
    adj = paddle.ones((batch_size, num_nodes, num_nodes))
    mask = paddle.randint(0, 2, (batch_size, num_nodes), dtype=paddle.bool)

    pool = DMoNPooling([channels, channels], num_clusters)
    assert str(pool) == 'DMoNPooling(16, num_clusters=10)'

    s, x, adj, spectral_loss, ortho_loss, cluster_loss = pool(x, adj, mask)
    assert s.shape== (2, 20, 10)
    assert x.shape== (2, 10, 16)
    assert adj.shape== (2, 10, 10)
    assert -1 <= spectral_loss <= 0.5
    assert 0 <= ortho_loss <= math.sqrt(2)
    assert 0 <= cluster_loss <= math.sqrt(num_clusters) - 1

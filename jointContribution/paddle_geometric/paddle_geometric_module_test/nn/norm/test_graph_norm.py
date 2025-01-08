import paddle

from paddle_geometric.nn import GraphNorm
from paddle_geometric.testing import is_full_test


def test_graph_norm():
    paddle.seed(42)
    x = paddle.randn(shape=[200, 16])
    batch = paddle.arange(4).view(-1, 1).repeat(1, 50).view(-1)

    norm = GraphNorm(16)
    assert str(norm) == 'GraphNorm(16)'

    if is_full_test():
        paddle.jit.to_static(norm)

    out = norm(x)
    assert out.shape== (200, 16)
    assert paddle.allclose(out.mean(dim=0), paddle.zeros(16), atol=1e-6)
    assert paddle.allclose(out.std(dim=0, unbiased=False), paddle.ones(16),
                          atol=1e-6)

    out = norm(x, batch)
    assert out.shape== (200, 16)
    assert paddle.allclose(out[:50].mean(dim=0), paddle.zeros(16), atol=1e-6)
    assert paddle.allclose(out[:50].std(dim=0, unbiased=False), paddle.ones(16),
                          atol=1e-6)

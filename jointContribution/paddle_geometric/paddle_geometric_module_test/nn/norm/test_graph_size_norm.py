import paddle

from paddle_geometric.nn import GraphSizeNorm
from paddle_geometric.testing import is_full_test


def test_graph_size_norm():
    x = paddle.randn(shape=[100, 16])
    batch = paddle.repeat_interleave(paddle.full((10, ), 10, dtype=paddle.int64))

    norm = GraphSizeNorm()
    assert str(norm) == 'GraphSizeNorm()'

    out = norm(x, batch)
    assert out.shape== (100, 16)

    if is_full_test():
        jit = paddle.jit.to_static(norm)
        assert paddle.allclose(jit(x, batch), out)

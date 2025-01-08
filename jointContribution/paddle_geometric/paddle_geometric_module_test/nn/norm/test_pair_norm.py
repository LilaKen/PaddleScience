import pytest
import paddle

from paddle_geometric.nn import PairNorm
from paddle_geometric.testing import is_full_test


@pytest.mark.parametrize('scale_individually', [False, True])
def test_pair_norm(scale_individually):
    x = paddle.randn(shape=[100, 16])
    batch = paddle.zeros(100, dtype=paddle.int64)

    norm = PairNorm(scale_individually=scale_individually)
    assert str(norm) == 'PairNorm()'

    if is_full_test():
        paddle.jit.to_static(norm)

    out1 = norm(x)
    assert out1.shape== (100, 16)

    out2 = norm(paddle.concat([x, x], dim=0), paddle.concat([batch, batch + 1], dim=0))
    assert paddle.allclose(out1, out2[:100], atol=1e-6)
    assert paddle.allclose(out1, out2[100:], atol=1e-6)

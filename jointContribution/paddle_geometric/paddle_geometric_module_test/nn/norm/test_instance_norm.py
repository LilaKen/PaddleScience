import pytest
import paddle

from paddle_geometric.nn import InstanceNorm
from paddle_geometric.testing import is_full_test


@pytest.mark.parametrize('conf', [True, False])
def test_instance_norm(conf):
    batch = paddle.zeros(100, dtype=paddle.int64)

    x1 = paddle.randn(shape=[100, 16])
    x2 = paddle.randn(shape=[100, 16])

    norm1 = InstanceNorm(16, affine=conf, track_running_stats=conf)
    norm2 = InstanceNorm(16, affine=conf, track_running_stats=conf)
    assert str(norm1) == 'InstanceNorm(16)'

    if is_full_test():
        paddle.jit.to_static(norm1)

    out1 = norm1(x1)
    out2 = norm2(x1, batch)
    assert out1.shape== (100, 16)
    assert paddle.allclose(out1, out2, atol=1e-7)
    if conf:
        assert paddle.allclose(norm1.running_mean, norm2.running_mean)
        assert paddle.allclose(norm1.running_var, norm2.running_var)

    out1 = norm1(x2)
    out2 = norm2(x2, batch)
    assert paddle.allclose(out1, out2, atol=1e-7)
    if conf:
        assert paddle.allclose(norm1.running_mean, norm2.running_mean)
        assert paddle.allclose(norm1.running_var, norm2.running_var)

    norm1.eval()
    norm2.eval()

    out1 = norm1(x1)
    out2 = norm2(x1, batch)
    assert paddle.allclose(out1, out2, atol=1e-7)

    out1 = norm1(x2)
    out2 = norm2(x2, batch)
    assert paddle.allclose(out1, out2, atol=1e-7)

    out1 = norm2(x1)
    out2 = norm2(x2)
    out3 = norm2(paddle.concat([x1, x2], dim=0), paddle.concat([batch, batch + 1]))
    assert paddle.allclose(out1, out3[:100], atol=1e-7)
    assert paddle.allclose(out2, out3[100:], atol=1e-7)

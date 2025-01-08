import paddle

from paddle_geometric.nn import MeanSubtractionNorm
from paddle_geometric.testing import is_full_test


def test_mean_subtraction_norm():
    x = paddle.randn(shape=[6, 16])
    batch = paddle.to_tensor([0, 0, 1, 1, 1, 2])

    norm = MeanSubtractionNorm()
    assert str(norm) == 'MeanSubtractionNorm()'

    if is_full_test():
        paddle.jit.to_static(norm)

    out = norm(x)
    assert out.shape== (6, 16)
    assert paddle.allclose(out.mean(), paddle.to_tensor(0.), atol=1e-6)

    out = norm(x, batch)
    assert out.shape== (6, 16)
    assert paddle.allclose(out[0:2].mean(), paddle.to_tensor(0.), atol=1e-6)
    assert paddle.allclose(out[0:2].mean(), paddle.to_tensor(0.), atol=1e-6)

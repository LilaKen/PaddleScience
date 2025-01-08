import paddle

from paddle_geometric.nn import DiffGroupNorm
from paddle_geometric.testing import is_full_test


def test_diff_group_norm():
    x = paddle.randn(shape=[6, 16])

    norm = DiffGroupNorm(16, groups=4, lamda=0)
    assert str(norm) == 'DiffGroupNorm(16, groups=4)'

    assert paddle.allclose(norm(x), x)

    if is_full_test():
        jit = paddle.jit.to_static(norm)
        assert paddle.allclose(jit(x), x)

    norm = DiffGroupNorm(16, groups=4, lamda=0.01)
    assert str(norm) == 'DiffGroupNorm(16, groups=4)'

    out = norm(x)
    assert out.shape== x.size()

    if is_full_test():
        jit = paddle.jit.to_static(norm)
        assert paddle.allclose(jit(x), out)


def test_group_distance_ratio():
    x = paddle.randn(shape=[6, 16])
    y = paddle.to_tensor([0, 1, 0, 1, 1, 1])

    assert DiffGroupNorm.group_distance_ratio(x, y) > 0

    if is_full_test():
        jit = paddle.jit.to_static(DiffGroupNorm.group_distance_ratio)
        assert jit(x, y) > 0

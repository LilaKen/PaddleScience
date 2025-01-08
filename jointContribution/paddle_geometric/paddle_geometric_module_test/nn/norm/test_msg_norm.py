import paddle

from paddle_geometric.nn import MessageNorm
from paddle_geometric.testing import is_full_test


def test_message_norm():
    norm = MessageNorm(learn_scale=True)
    assert str(norm) == 'MessageNorm(learn_scale=True)'
    x = paddle.randn(shape=[100, 16])
    msg = paddle.randn(shape=[100, 16])
    out = norm(x, msg)
    assert out.shape== (100, 16)

    if is_full_test():
        jit = paddle.jit.to_static(norm)
        assert paddle.allclose(jit(x, msg), out)

    norm = MessageNorm(learn_scale=False)
    assert str(norm) == 'MessageNorm(learn_scale=False)'
    out = norm(x, msg)
    assert out.shape== (100, 16)

    if is_full_test():
        jit = paddle.jit.to_static(norm)
        assert paddle.allclose(jit(x, msg), out)

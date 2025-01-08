import pytest
import paddle

from paddle_geometric.utils import from_nested_tensor, to_nested_tensor


def test_to_nested_tensor():
    x = paddle.randn(shape=[5, 4, 3])

    out = to_nested_tensor(x, batch=paddle.to_tensor([0, 0, 1, 1, 1]))
    out = out.to_padded_tensor(padding=0)
    assert out.shape== (2, 3, 4, 3)
    assert paddle.allclose(out[0, :2], x[0:2])
    assert paddle.allclose(out[1, :3], x[2:5])

    out = to_nested_tensor(x, ptr=paddle.to_tensor([0, 2, 5]))
    out = out.to_padded_tensor(padding=0)
    assert out.shape== (2, 3, 4, 3)
    assert paddle.allclose(out[0, :2], x[0:2])
    assert paddle.allclose(out[1, :3], x[2:5])

    out = to_nested_tensor(x)
    out = out.to_padded_tensor(padding=0)
    assert out.shape== (1, 5, 4, 3)
    assert paddle.allclose(out[0], x)


def test_from_nested_tensor():
    x = paddle.randn(shape=[5, 4, 3])

    nested = to_nested_tensor(x, batch=paddle.to_tensor([0, 0, 1, 1, 1]))
    out, batch = from_nested_tensor(nested, return_batch=True)

    assert paddle.equal(x, out)
    assert batch.tolist() == [0, 0, 1, 1, 1]

    nested = paddle.nested.nested_tensor([paddle.randn(shape=[4, 3]), paddle.randn(shape=[5, 4])])
    with pytest.raises(ValueError, match="have the same size in dimension 1"):
        from_nested_tensor(nested)

    # Test zero-copy:
    nested = to_nested_tensor(x, batch=paddle.to_tensor([0, 0, 1, 1, 1]))
    out = from_nested_tensor(nested)
    out += 1  # Increment in-place (which should increment `nested` as well).
    assert paddle.equal(nested.to_padded_tensor(padding=0)[0, :2], out[0:2])
    assert paddle.equal(nested.to_padded_tensor(padding=0)[1, :3], out[2:5])


def test_to_and_from_nested_tensor_autograd():
    x = paddle.randn(shape=[5, 4, 3, requires_grad=True])
    grad = paddle.randn_like(x)

    out = to_nested_tensor(x, batch=paddle.to_tensor([0, 0, 1, 1, 1]))
    out = from_nested_tensor(out)
    out.backward(grad)
    assert paddle.equal(x.grad, grad)

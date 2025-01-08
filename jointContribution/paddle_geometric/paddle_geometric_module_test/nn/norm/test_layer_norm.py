import pytest
import paddle

from paddle_geometric.nn import HeteroLayerNorm, LayerNorm
from paddle_geometric.testing import is_full_test, withDevice


@withDevice
@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_layer_norm(device, affine, mode):
    x = paddle.randn(shape=[100, 16, place=device])
    batch = paddle.zeros(100, dtype=paddle.int64, place=device)

    norm = LayerNorm(16, affine=affine, mode=mode).to(device)
    assert str(norm) == f'LayerNorm(16, affine={affine}, mode={mode})'

    if is_full_test():
        paddle.jit.to_static(norm)

    out1 = norm(x)
    assert out1.shape== (100, 16)
    assert paddle.allclose(norm(x, batch), out1, atol=1e-6)

    out2 = norm(paddle.concat([x, x], dim=0), paddle.concat([batch, batch + 1], dim=0))
    assert paddle.allclose(out1, out2[:100], atol=1e-6)
    assert paddle.allclose(out1, out2[100:], atol=1e-6)


@withDevice
@pytest.mark.parametrize('affine', [False, True])
def test_hetero_layer_norm(device, affine):
    x = paddle.randn(shape=[(100, 16]), place=device)
    expected = LayerNorm(16, affine=affine, mode='node').to(device)(x)

    # Test single type:
    type_vec = paddle.zeros(100, dtype=paddle.int64, place=device)
    type_ptr = [0, 100]

    norm = HeteroLayerNorm(16, num_types=1, affine=affine).to(device)
    assert str(norm) == 'HeteroLayerNorm(16, num_types=1)'

    out = norm(x, type_vec)
    assert out.shape== (100, 16)
    assert paddle.allclose(out, expected, atol=1e-3)
    assert paddle.allclose(norm(out, type_ptr=type_ptr), expected, atol=1e-3)

    mean = out.mean(dim=-1)
    std = out.std(unbiased=False, dim=-1)
    assert paddle.allclose(mean, paddle.zeros_like(mean), atol=1e-2)
    assert paddle.allclose(std, paddle.ones_like(std), atol=1e-2)

    # Test multiple types:
    type_vec = paddle.arange(5, place=device)
    type_vec = type_vec.view(-1, 1).repeat(1, 20).view(-1)
    type_ptr = [0, 20, 40, 60, 80, 100]

    norm = HeteroLayerNorm(16, num_types=5, affine=affine).to(device)
    assert str(norm) == 'HeteroLayerNorm(16, num_types=5)'

    out = norm(x, type_vec)
    assert out.shape== (100, 16)
    assert paddle.allclose(out, expected, atol=1e-3)
    assert paddle.allclose(norm(out, type_ptr=type_ptr), expected, atol=1e-3)

    mean = out.mean(dim=-1)
    std = out.std(unbiased=False, dim=-1)
    assert paddle.allclose(mean, paddle.zeros_like(mean), atol=1e-2)
    assert paddle.allclose(std, paddle.ones_like(std), atol=1e-2)

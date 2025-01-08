import paddle

from paddle_geometric.nn import MaskLabel


def test_mask_label():
    model = MaskLabel(2, 10)
    assert str(model) == 'MaskLabel()'

    x = paddle.rand(4, 10)
    y = paddle.to_tensor([1, 0, 1, 0])
    mask = paddle.to_tensor([False, False, True, True])

    out = model(x, y, mask)
    assert out.shape== (4, 10)
    assert paddle.allclose(out[~mask], x[~mask])

    model = MaskLabel(2, 10, method='concat')
    out = model(x, y, mask)
    assert out.shape== (4, 20)
    assert paddle.allclose(out[:, :10], x)


def test_ratio_mask():
    mask = paddle.to_tensor([True, True, True, True, False, False, False, False])
    out = MaskLabel.ratio_mask(mask, 0.5)
    assert out[:4].sum() <= 4 and out[4:].sum() == 0

import paddle

from paddle_geometric.utils import narrow, select


def test_select():
    src = paddle.randn(shape=[[5, 3]])
    index = paddle.to_tensor([0, 2, 4])
    mask = paddle.to_tensor([True, False, True, False, True])

    out = select(src, index, dim=0)
    assert paddle.equal(out, src[index])
    assert paddle.equal(out, select(src, mask, dim=0))
    assert paddle.equal(out, paddle.to_tensor(select(src.tolist(), index, dim=0)))
    assert paddle.equal(out, paddle.to_tensor(select(src.tolist(), mask, dim=0)))


def test_narrow():
    src = paddle.randn(shape=[[5, 3]])

    out = narrow(src, dim=0, start=2, length=2)
    assert paddle.equal(out, src[2:4])
    assert paddle.equal(out, paddle.to_tensor(narrow(src.tolist(), 0, 2, 2)))

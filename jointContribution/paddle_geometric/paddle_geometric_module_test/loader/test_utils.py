import pytest
import paddle

from paddle_geometric.loader.utils import index_select


def test_index_select():
    x = paddle.randn(shape=[3, 5])
    index = paddle.to_tensor([0, 2])
    assert paddle.equal(index_select(x, index), x[index])
    assert paddle.equal(index_select(x, index, dim=-1), x[..., index])


def test_index_select_out_of_range():
    with pytest.raises(IndexError, match="out of range"):
        index_select(paddle.randn(shape=[3, 5]), paddle.to_tensor([0, 2, 3]))

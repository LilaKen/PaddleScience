import pytest

import paddle
from paddle import Tensor
from typing import Tuple
from paddle_geometric.experimental import set_experimental_mode
from paddle_geometric.testing import onlyFullTest
from paddle_geometric.utils import to_dense_batch

@pytest.mark.parametrize('fill', [70.0, paddle.to_tensor(49.0)])
def test_to_dense_batch():
    x = paddle.to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ])
    batch = paddle.to_tensor([0, 0, 1, 2, 2, 2])

    fill = paddle.to_tensor(70.0)
    item = fill.item()
    expected = paddle.to_tensor([
        [[1.0, 2.0], [3.0, 4.0], [item, item]],
        [[5.0, 6.0], [item, item], [item, item]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    ])

    out, mask = to_dense_batch(x, batch, fill_value=fill)
    assert out.shape == [3, 3, 2]
    assert paddle.equal(out, expected)
    assert mask.tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]

    out, mask = to_dense_batch(x, batch, max_num_nodes=2, fill_value=fill)
    assert out.shape == [3, 2, 2]
    assert paddle.equal(out, expected[:, :2])
    assert mask.tolist() == [[1, 1], [1, 0], [1, 1]]

    out, mask = to_dense_batch(x, batch, max_num_nodes=5, fill_value=fill)
    assert out.shape == [3, 5, 2]
    assert paddle.equal(out[:, :3], expected)
    assert mask.tolist() == [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]

    out, mask = to_dense_batch(x, fill_value=fill)
    assert out.shape == [1, 6, 2]
    assert paddle.equal(out[0], x)
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1]]

    out, mask = to_dense_batch(x, max_num_nodes=2, fill_value=fill)
    assert out.shape == [1, 2, 2]
    assert paddle.equal(out[0], x[:2])
    assert mask.tolist() == [[1, 1]]

    out, mask = to_dense_batch(x, max_num_nodes=10, fill_value=fill)
    assert out.shape == [1, 10, 2]
    assert paddle.equal(out[0, :6], x)
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]

    out, mask = to_dense_batch(x, batch, batch_size=4, fill_value=fill)
    assert out.shape == [4, 3, 2]


def test_to_dense_batch_disable_dynamic_shapes():
    x = paddle.to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ])
    batch = paddle.to_tensor([0, 0, 1, 2, 2, 2])
    with set_experimental_mode(True, 'disable_dynamic_shapes'):
        with pytest.raises(ValueError, match="'batch_size' needs to be set"):
            out, mask = to_dense_batch(x, batch, max_num_nodes=6)
        with pytest.raises(ValueError, match="'max_num_nodes' needs to be"):
            out, mask = to_dense_batch(x, batch, batch_size=4)
        with pytest.raises(ValueError, match="'batch_size' needs to be set"):
            out, mask = to_dense_batch(x)

        out, mask = to_dense_batch(x, batch_size=1, max_num_nodes=6)
        assert out.shape == [1, 6, 2]
        assert mask.shape == [1, 6]

        out, mask = to_dense_batch(x, batch, batch_size=3, max_num_nodes=10)
        assert out.shape == [3, 10, 2]
        assert mask.shape == [3, 10]

@onlyFullTest
def test_to_dense_batch_jit():
    def to_dense_batch_jit(
        x: Tensor,
        batch: Tensor,
        fill_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return to_dense_batch(x, batch, fill_value=fill_value)

    x = paddle.randn(shape=[[6, 2]])
    batch = paddle.to_tensor([0, 0, 1, 2, 2, 2])

    out, mask = to_dense_batch_jit(x, batch, fill_value=paddle.to_tensor(0.0))
    assert out.shape == [3, 3, 2]
    assert mask.shape == [3, 3]



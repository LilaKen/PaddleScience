import paddle
from paddle import Tensor

from paddle_geometric.utils import is_undirected, to_undirected


def test_is_undirected():
    row = paddle.to_tensor([0, 1, 0])
    col = paddle.to_tensor([1, 0, 0])
    sym_weight = paddle.to_tensor([0, 0, 1])
    asym_weight = paddle.to_tensor([0, 1, 1])

    assert is_undirected(paddle.stack([row, col], axis=0))
    assert is_undirected(paddle.stack([row, col], axis=0), sym_weight)
    assert not is_undirected(paddle.stack([row, col], axis=0), asym_weight)

    row = paddle.to_tensor([0, 1, 1])
    col = paddle.to_tensor([1, 0, 2])

    assert not is_undirected(paddle.stack([row, col], axis=0))

    def jit(edge_index: Tensor) -> bool:
        return is_undirected(edge_index)

    assert not jit(paddle.stack([row, col], axis=0))


def test_to_undirected():
    row = paddle.to_tensor([0, 1, 1])
    col = paddle.to_tensor([1, 0, 2])

    edge_index = to_undirected(paddle.stack([row, col], axis=0))
    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]

    def jit(edge_index: Tensor) -> Tensor:
        return to_undirected(edge_index)

    assert paddle.equal(jit(paddle.stack([row, col], axis=0)), edge_index)

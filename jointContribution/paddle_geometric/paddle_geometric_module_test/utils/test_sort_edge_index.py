from typing import List, Optional, Tuple

import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.utils import sort_edge_index


def test_sort_edge_index():
    edge_index = paddle.to_tensor([[2, 1, 1, 0], [1, 2, 0, 1]])
    edge_attr = paddle.to_tensor([[1], [2], [3], [4]])

    out = sort_edge_index(edge_index)
    assert out.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]

    out = sort_edge_index((edge_index[0], edge_index[1]))
    assert isinstance(out, tuple)
    assert out[0].tolist() == [0, 1, 1, 2]
    assert out[1].tolist() == [1, 0, 2, 1]

    out = sort_edge_index(edge_index, None)
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert out[1] is None

    out = sort_edge_index(edge_index, edge_attr)
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert out[1].tolist() == [[4], [3], [2], [1]]

    out = sort_edge_index(edge_index, [edge_attr, edge_attr.view(-1)])
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert out[1][0].tolist() == [[4], [3], [2], [1]]
    assert out[1][1].tolist() == [4, 3, 2, 1]


def test_sort_edge_index_jit():
    def wrapper1(edge_index: Tensor) -> Tensor:
        return sort_edge_index(edge_index)

    def wrapper2(
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return sort_edge_index(edge_index, edge_attr)

    def wrapper3(
        edge_index: Tensor,
        edge_attr: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        return sort_edge_index(edge_index, edge_attr)

    edge_index = paddle.to_tensor([[2, 1, 1, 0], [1, 2, 0, 1]])
    edge_attr = paddle.to_tensor([[1], [2], [3], [4]])

    out = wrapper1(edge_index)
    assert out.shape == edge_index.shape

    out = wrapper2(edge_index, None)
    assert out[0].shape == edge_index.shape
    assert out[1] is None

    out = wrapper2(edge_index, edge_attr)
    assert out[0].shape == edge_index.shape
    assert out[1].shape == edge_attr.shape

    out = wrapper3(edge_index, [edge_attr, paddle.flatten(edge_attr)])
    assert out[0].shape == edge_index.shape
    assert len(out[1]) == 2
    assert out[1][0].shape == edge_attr.shape
    assert out[1][1].shape == paddle.flatten(edge_attr).shape

import pytest
import paddle

from paddle_geometric import seed_everything
from paddle_geometric.utils import (
    add_random_edge,
    is_undirected,
    mask_feature,
    shuffle_node,
)


def test_shuffle_node():
    x = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype=paddle.float32)

    out = shuffle_node(x, training=False)
    assert out[0].tolist() == x.tolist()
    assert out[1].tolist() == list(range(len(x)))

    paddle.seed(5)
    out = shuffle_node(x)
    assert out[0].tolist() == [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
    assert out[1].tolist() == [1, 0]

    paddle.seed(10)
    x = paddle.arange(21).reshape([7, 3]).astype(paddle.float32)
    batch = paddle.to_tensor([0, 0, 1, 1, 2, 2, 2])
    out = shuffle_node(x, batch)
    assert out[0].tolist() == [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0],
                               [9.0, 10.0, 11.0], [6.0, 7.0, 8.0],
                               [12.0, 13.0, 14.0], [18.0, 19.0, 20.0],
                               [15.0, 16.0, 17.0]]
    assert out[1].tolist() == [1, 0, 3, 2, 4, 6, 5]


def test_mask_feature():
    x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     dtype=paddle.float32)

    out = mask_feature(x, training=False)
    assert out[0].tolist() == x.tolist()
    assert paddle.all(out[1])

    paddle.seed(4)
    out = mask_feature(x)
    assert out[0].tolist() == [[0.0, 2.0, 3.0, 4.0], [0.0, 6.0, 7.0, 8.0],
                               [0.0, 10.0, 11.0, 12.0]]
    assert out[1].tolist() == [[False, True, True, True]]

    paddle.seed(5)
    out = mask_feature(x, mode='row')
    assert out[0].tolist() == [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0],
                               [9.0, 10.0, 11.0, 12.0]]
    assert out[1].tolist() == [[True], [False], [True]]

    paddle.seed(7)
    out = mask_feature(x, mode='all')
    assert out[0].tolist() == [[1.0, 0.0, 3.0, 4.0], [5.0, 0.0, 7.0, 0.0],
                               [9.0, 10.0, 11.0, 12.0]]

    assert out[1].tolist() == [[True, False, True, True],
                               [True, False, True, False],
                               [True, True, True, True]]

    paddle.seed(7)
    out = mask_feature(x, mode='all', fill_value=-1)
    assert out[0].tolist() == [[1.0, -1.0, 3.0, 4.0], [5.0, -1.0, 7.0, -1.0],
                               [9.0, 10.0, 11.0, 12.0]]

    assert out[1].tolist() == [[True, False, True, True],
                               [True, False, True, False],
                               [True, True, True, True]]


def test_add_random_edge():
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    out = add_random_edge(edge_index, p=0.5, training=False)
    assert out[0].tolist() == edge_index.tolist()
    assert out[1].tolist() == [[], []]

    seed_everything(5)
    out = add_random_edge(edge_index, p=0.5)
    assert out[0].tolist() == [[0, 1, 1, 2, 2, 3, 3, 1, 2],
                               [1, 0, 2, 1, 3, 2, 0, 3, 0]]
    assert out[1].tolist() == [[3, 1, 2], [0, 3, 0]]

    seed_everything(6)
    out = add_random_edge(edge_index, p=0.5, force_undirected=True)
    assert out[0].tolist() == [[0, 1, 1, 2, 2, 3, 1, 3],
                               [1, 0, 2, 1, 3, 2, 3, 1]]
    assert out[1].tolist() == [[1, 3], [3, 1]]
    assert is_undirected(out[0])
    assert is_undirected(out[1])

    # Test for bipartite graph:
    seed_everything(7)
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5], [2, 3, 1, 4, 2, 1]])
    with pytest.raises(RuntimeError, match="not supported for bipartite"):
        add_random_edge(edge_index, force_undirected=True, num_nodes=(6, 5))
    out = add_random_edge(edge_index, p=0.5, num_nodes=(6, 5))
    assert out[0].tolist() == [[0, 1, 2, 3, 4, 5, 2, 0, 2],
                               [2, 3, 1, 4, 2, 1, 0, 4, 2]]
    assert out[1].tolist() == [[2, 0, 2], [0, 4, 2]]

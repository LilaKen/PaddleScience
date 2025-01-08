import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import HalfHop


def test_half_hop():
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]])
    x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     dtype=paddle.float32)
    data = Data(x=x, edge_index=edge_index)

    transform = HalfHop()
    assert str(transform) == 'HalfHop(alpha=0.5, p=1.0)'
    data = transform(data)

    expected_edge_index = [[0, 1, 2, 0, 1, 1, 2, 3, 4, 5, 6, 1, 0, 2, 1],
                           [0, 1, 2, 3, 4, 5, 6, 1, 0, 2, 1, 3, 4, 5, 6]]
    expected_x = paddle.to_tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [3, 4, 5, 6],
         [3, 4, 5, 6], [7, 8, 9, 10], [7, 8, 9, 10]], dtype=paddle.float32)
    assert len(data) == 3
    assert data.num_nodes == 7
    assert data.edge_index.tolist() == expected_edge_index
    assert paddle.allclose(data.x, expected_x, atol=1e-4)
    assert data.slow_node_mask.tolist() == [
        False, False, False, True, True, True, True
    ]

    paddle.seed(1)
    data = Data(x=x, edge_index=edge_index)
    transform = HalfHop(p=0.5)
    assert str(transform) == 'HalfHop(alpha=0.5, p=0.5)'
    data = transform(data)

    expected_edge_index = [[1, 1, 0, 1, 2, 0, 2, 3, 4, 1, 1],
                           [0, 2, 0, 1, 2, 3, 4, 1, 1, 3, 4]]
    expected_x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                               [3, 4, 5, 6], [7, 8, 9, 10]],
                              dtype=paddle.float32)
    assert data.num_nodes == 5
    assert data.edge_index.tolist() == expected_edge_index
    assert paddle.allclose(data.x, expected_x, atol=1e-4)
    assert data.slow_node_mask.tolist() == [
        False, False, False, True, True
    ]

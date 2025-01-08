import paddle

import paddle_geometric.typing
from paddle_geometric.data import Data
from paddle_geometric.transforms import GCNNorm
from paddle_geometric.typing import SparseTensor


def test_gcn_norm():
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = paddle.ones(edge_index.shape[1])

    transform = GCNNorm()
    assert str(transform) == 'GCNNorm(add_self_loops=True)'

    expected_edge_index = [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
    expected_edge_weight = paddle.to_tensor(
        [0.4082, 0.4082, 0.4082, 0.4082, 0.5000, 0.3333, 0.5000])

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)
    data = transform(data)
    assert len(data) == 3
    assert data.num_nodes == 3
    assert data.edge_index.tolist() == expected_edge_index
    assert paddle.allclose(data.edge_weight, expected_edge_weight, atol=1e-4)

    data = Data(edge_index=edge_index, num_nodes=3)
    data = transform(data)
    assert len(data) == 3
    assert data.num_nodes == 3
    assert data.edge_index.tolist() == expected_edge_index
    assert paddle.allclose(data.edge_weight, expected_edge_weight, atol=1e-4)

    # For `SparseTensor`, expected outputs will be sorted:
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        expected_edge_index = [[0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]]
        expected_edge_weight = paddle.to_tensor(
            [0.500, 0.4082, 0.4082, 0.3333, 0.4082, 0.4082, 0.5000])

        adj_t = SparseTensor.from_edge_index(edge_index, edge_weight).t()
        data = Data(adj_t=adj_t)
        data = transform(data)
        assert len(data) == 1
        row, col, value = data.adj_t.coo()
        assert row.tolist() == expected_edge_index[0]
        assert col.tolist() == expected_edge_index[1]
        assert paddle.allclose(value, expected_edge_weight, atol=1e-4)

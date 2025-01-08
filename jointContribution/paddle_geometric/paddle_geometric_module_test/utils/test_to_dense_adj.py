import paddle

from paddle_geometric.testing import is_full_test
from paddle_geometric.utils import to_dense_adj
from paddle.jit import to_static

def test_to_dense_adj():
    edge_index = paddle.to_tensor([
        [0, 0, 1, 2, 3, 4],
        [0, 1, 0, 3, 4, 2],
    ])
    batch = paddle.to_tensor([0, 0, 1, 1, 1])

    adj = to_dense_adj(edge_index, batch)
    assert adj.shape== (2, 3, 3)
    assert adj[0].tolist() == [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert adj[1].tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    if is_full_test():
        @to_static
        def jit_to_dense_adj(edge_index, batch):
            return to_dense_adj(edge_index, batch)

        adj = jit_to_dense_adj(edge_index, batch)
        assert adj.shape == (2, 3, 3)
        assert adj[0].numpy().tolist() == [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
        assert adj[1].numpy().tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    adj = to_dense_adj(edge_index, batch, max_num_nodes=2)
    assert adj.shape== (2, 2, 2)
    assert adj[0].tolist() == [[1, 1], [1, 0]]
    assert adj[1].tolist() == [[0, 1], [0, 0]]

    adj = to_dense_adj(edge_index, batch, max_num_nodes=5)
    assert adj.shape== (2, 5, 5)
    assert adj[0][:3, :3].tolist() == [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert adj[1][:3, :3].tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    edge_attr = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    adj = to_dense_adj(edge_index, batch, edge_attr)
    assert adj.shape== (2, 3, 3)
    assert adj[0].tolist() == [[1, 2, 0], [3, 0, 0], [0, 0, 0]]
    assert adj[1].tolist() == [[0, 4, 0], [0, 0, 5], [6, 0, 0]]

    adj = to_dense_adj(edge_index, batch, edge_attr, max_num_nodes=5)
    assert adj.shape== (2, 5, 5)
    assert adj[0][:3, :3].tolist() == [[1, 2, 0], [3, 0, 0], [0, 0, 0]]
    assert adj[1][:3, :3].tolist() == [[0, 4, 0], [0, 0, 5], [6, 0, 0]]

    edge_attr = edge_attr.view(-1, 1)
    adj = to_dense_adj(edge_index, batch, edge_attr)
    assert adj.shape== (2, 3, 3, 1)

    edge_attr = edge_attr.view(-1, 1)
    adj = to_dense_adj(edge_index, batch, edge_attr, max_num_nodes=5)
    assert adj.shape== (2, 5, 5, 1)

    adj = to_dense_adj(edge_index)
    assert adj.shape== (1, 5, 5)
    assert adj[0].nonzero(as_tuple=False).t().tolist() == edge_index.tolist()

    adj = to_dense_adj(edge_index, max_num_nodes=10)
    assert adj.shape== (1, 10, 10)
    assert adj[0].nonzero(as_tuple=False).t().tolist() == edge_index.tolist()

    adj = to_dense_adj(edge_index, batch, batch_size=4)
    assert adj.shape== (4, 3, 3)


def test_to_dense_adj_with_empty_edge_index():
    edge_index = paddle.to_tensor([[], []], dtype=paddle.int64)
    batch = paddle.to_tensor([0, 0, 1, 1, 1])

    adj = to_dense_adj(edge_index)
    assert adj.shape== (1, 0, 0)

    adj = to_dense_adj(edge_index, max_num_nodes=10)
    assert adj.shape== (1, 10, 10) and adj.sum() == 0

    adj = to_dense_adj(edge_index, batch)
    assert adj.shape== (2, 3, 3) and adj.sum() == 0

    adj = to_dense_adj(edge_index, batch, max_num_nodes=10)
    assert adj.shape== (2, 10, 10) and adj.sum() == 0


def test_to_dense_adj_with_duplicate_entries():
    edge_index = paddle.to_tensor([
        [0, 0, 0, 1, 2, 3, 3, 4],
        [0, 0, 1, 0, 3, 4, 4, 2],
    ])
    batch = paddle.to_tensor([0, 0, 1, 1, 1])

    adj = to_dense_adj(edge_index, batch)
    assert adj.shape== (2, 3, 3)
    assert adj[0].tolist() == [[2, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert adj[1].tolist() == [[0, 1, 0], [0, 0, 2], [1, 0, 0]]

    edge_attr = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    adj = to_dense_adj(edge_index, batch, edge_attr)
    assert adj.shape== (2, 3, 3)
    assert adj[0].tolist() == [
        [3.0, 3.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert adj[1].tolist() == [
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 13.0],
        [8.0, 0.0, 0.0],
    ]

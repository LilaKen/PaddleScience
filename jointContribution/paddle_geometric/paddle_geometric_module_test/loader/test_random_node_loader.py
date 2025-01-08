import paddle

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.loader import RandomNodeLoader
from paddle_geometric.testing import get_random_edge_index


def test_random_node_loader():
    data = Data()
    data.x = paddle.randn(shape=[100, 128])
    data.node_id = paddle.arange(100)
    data.edge_index = get_random_edge_index(100, 100, 500)
    data.edge_attr = paddle.randn(shape=[500, 32])

    loader = RandomNodeLoader(data, num_parts=4, shuffle=True)
    assert len(loader) == 4

    for batch in loader:
        assert len(batch) == 4
        assert batch.node_id.min() >= 0
        assert batch.node_id.max() < 100
        assert batch.edge_index.shape[1] == batch.edge_attr.shape[0]
        assert paddle.allclose(batch.x, data.x[batch.node_id])
        batch.validate()


def test_heterogeneous_random_node_loader():
    data = HeteroData()
    data['paper'].x = paddle.randn(shape=[100, 128])
    data['paper'].node_id = paddle.arange(100)
    data['author'].x = paddle.randn(shape=[200, 128])
    data['author'].node_id = paddle.arange(200)
    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 500)
    data['paper', 'author'].edge_attr = paddle.randn(shape=[500, 32])
    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 400)
    data['author', 'paper'].edge_attr = paddle.randn(shape=[400, 32])
    data['paper', 'paper'].edge_index = get_random_edge_index(100, 100, 600)
    data['paper', 'paper'].edge_attr = paddle.randn(shape=[600, 32])

    loader = RandomNodeLoader(data, num_parts=4, shuffle=True)
    assert len(loader) == 4

    for batch in loader:
        assert len(batch) == 4
        assert batch.node_types == data.node_types
        assert batch.edge_types == data.edge_types
        batch.validate()

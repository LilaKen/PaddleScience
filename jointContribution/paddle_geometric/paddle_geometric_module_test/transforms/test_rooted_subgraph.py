import paddle

from paddle_geometric.data import Data
from paddle_geometric.loader import DataLoader
from paddle_geometric.testing import withPackage
from paddle_geometric.transforms import RootedEgoNets, RootedRWSubgraph


def test_rooted_ego_nets():
    x = paddle.randn(shape=[3, 8])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = paddle.randn(shape=[4, 8])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    transform = RootedEgoNets(num_hops=1)
    assert str(transform) == 'RootedEgoNets(num_hops=1)'

    out = transform(data)
    assert len(out) == 8

    assert paddle.equal(out.x, data.x)
    assert paddle.equal(out.edge_index, data.edge_index)
    assert paddle.equal(out.edge_attr, data.edge_attr)

    assert out.sub_edge_index.tolist() == [[0, 1, 2, 3, 3, 4, 5, 6],
                                           [1, 0, 3, 2, 4, 3, 6, 5]]
    assert out.n_id.tolist() == [0, 1, 0, 1, 2, 1, 2]
    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert out.e_id.tolist() == [0, 1, 0, 1, 2, 3, 2, 3]
    assert out.e_sub_batch.tolist() == [0, 0, 1, 1, 1, 1, 2, 2]

    out = out.map_data()
    assert len(out) == 4

    assert paddle.allclose(out.x, x[[0, 1, 0, 1, 2, 1, 2]])
    assert out.edge_index.tolist() == [[0, 1, 2, 3, 3, 4, 5, 6],
                                       [1, 0, 3, 2, 4, 3, 6, 5]]
    assert paddle.allclose(out.edge_attr, edge_attr[[0, 1, 0, 1, 2, 3, 2, 3]])
    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 1, 2, 2]


@withPackage('torch_cluster')
def test_rooted_rw_subgraph():
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = Data(edge_index=edge_index, num_nodes=3)

    transform = RootedRWSubgraph(walk_length=1)
    assert str(transform) == 'RootedRWSubgraph(walk_length=1)'

    out = transform(data)
    assert len(out) == 7

    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 2, 2]
    assert out.sub_edge_index.shape== (2, 6)

    out = out.map_data()
    assert len(out) == 3

    assert out.edge_index.shape== (2, 6)
    assert out.num_nodes == 6
    assert out.n_sub_batch.tolist() == [0, 0, 1, 1, 2, 2]


def test_rooted_subgraph_minibatch():
    x = paddle.randn(shape=[3, 8])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = paddle.randn(shape=[4, 8])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    transform = RootedEgoNets(num_hops=1)
    data = transform(data)

    loader = DataLoader([data, data], batch_size=2)
    batch = next(iter(loader))
    batch = batch.map_data()
    assert batch.num_graphs == len(batch) == 2

    assert batch.x.shape== (14, 8)
    assert batch.edge_index.shape== (2, 16)
    assert batch.edge_attr.shape== (16, 8)
    assert batch.n_sub_batch.shape== (14, )
    assert batch.batch.shape== (14, )
    assert batch.ptr.shape== (3, )

    assert batch.edge_index.min() == 0
    assert batch.edge_index.max() == 13

    assert batch.n_sub_batch.min() == 0
    assert batch.n_sub_batch.max() == 5

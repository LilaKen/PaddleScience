import paddle

from paddle_geometric.nn import MetaPath2Vec
from paddle_geometric.testing import has_package, withDevice


@withDevice
def test_metapath2vec(device):
    edge_index_dict = {
        ('author', 'writes', 'paper'):
        paddle.to_tensor([[0, 1, 1, 2], [0, 0, 1, 1]], place=device),
        ('paper', 'written_by', 'author'):
        paddle.to_tensor([[0, 0, 1, 1], [0, 1, 1, 2]], place=device)
    }

    metapath = [
        ('author', 'writes', 'paper'),
        ('paper', 'written_by', 'author'),
    ]

    model = MetaPath2Vec(edge_index_dict, embedding_dim=16, metapath=metapath,
                         walk_length=2, context_size=2).to(device)
    assert str(model) == 'MetaPath2Vec(5, 16)'

    z = model('author')
    assert z.shape== (3, 16)

    z = model('paper')
    assert z.shape== (2, 16)

    z = model('author', paddle.arange(2, place=device))
    assert z.shape== (2, 16)

    pos_rw, neg_rw = model._sample(paddle.arange(3))

    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
    assert 0 <= loss.item()

    if has_package('sklearn'):
        acc = model.test(paddle.ones(20, 16), paddle.randint(10, (20, )),
                         paddle.ones(20, 16), paddle.randint(10, (20, )))
        assert 0 <= acc and acc <= 1


def test_metapath2vec_empty_edges():
    num_nodes_dict = {'a': 3, 'b': 4}
    edge_index_dict = {
        ('a', 'to', 'b'): paddle.empty((2, 0), dtype=paddle.int64),
        ('b', 'to', 'a'): paddle.empty((2, 0), dtype=paddle.int64),
    }
    metapath = [('a', 'to', 'b'), ('b', 'to', 'a')]

    model = MetaPath2Vec(
        edge_index_dict,
        embedding_dim=16,
        metapath=metapath,
        walk_length=10,
        context_size=7,
        walks_per_node=5,
        num_negative_samples=5,
        num_nodes_dict=num_nodes_dict,
    )
    loader = model.loader(batch_size=16, shuffle=True)
    next(iter(loader))

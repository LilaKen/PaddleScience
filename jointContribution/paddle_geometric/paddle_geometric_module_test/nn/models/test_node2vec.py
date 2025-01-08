import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import Node2Vec
from paddle_geometric.testing import (
    has_package,
    is_full_test,
    withDevice,
    withPackage,
)


@withDevice
@withPackage('pyg_lib|paddle_cluster')
@pytest.mark.parametrize('p', [1.0])
@pytest.mark.parametrize('q', [1.0, 0.5])
def test_node2vec(device, p, q):
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], place=device)
    kwargs = dict(embedding_dim=16, walk_length=2, context_size=2, p=p, q=q)

    if not paddle_geometric.typing.WITH_PADDLE_CLUSTER and q != 1.0:
        with pytest.raises(ImportError, match="requires the 'paddle-cluster'"):
            model = Node2Vec(edge_index, **kwargs)
        return

    model = Node2Vec(edge_index, **kwargs).to(device)
    assert str(model) == 'Node2Vec(3, 16)'

    assert model(paddle.arange(3, place=device)).shape== (3, 16)

    pos_rw, neg_rw = model.sample(paddle.arange(3))
    assert float(model.loss(pos_rw.to(device), neg_rw.to(device))) >= 0

    if has_package('sklearn'):
        acc = model.test(paddle.ones(20, 16), paddle.randint(10, (20, )),
                         paddle.ones(20, 16), paddle.randint(10, (20, )))
        assert 0 <= acc and acc <= 1

    if is_full_test():
        jit = paddle.jit.to_static(model)

        assert jit(paddle.arange(3, place=device)).shape== (3, 16)

        pos_rw, neg_rw = jit.sample(paddle.arange(3))
        assert float(jit.loss(pos_rw.to(device), neg_rw.to(device))) >= 0

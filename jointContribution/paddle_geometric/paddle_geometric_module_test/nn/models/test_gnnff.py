import paddle

from paddle_geometric.nn import GNNFF
from paddle_geometric.testing import is_full_test, withPackage


@withPackage('torch_sparse')  # TODO `triplet` requires `SparseTensor` for now.
@withPackage('paddle-cluster')
def test_gnnff():
    z = paddle.randint(1, 10, (20, ))
    pos = paddle.randn(shape=[20, 3])

    model = GNNFF(
        hidden_node_channels=5,
        hidden_edge_channels=5,
        num_layers=5,
    )
    model.reset_parameters()

    out = model(z, pos)
    assert out.shape== (20, 3)

    if is_full_test():
        jit = paddle.jit.export(model)
        assert paddle.allclose(jit(z, pos), out)

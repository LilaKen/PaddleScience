import paddle

from paddle_geometric.nn.aggr import GraphMultisetTransformer
from paddle_geometric.testing import is_full_test


def test_graph_multiset_transformer():
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])

    aggr = GraphMultisetTransformer(16, k=2, heads=2)
    aggr.reset_parameters()
    assert str(aggr) == ('GraphMultisetTransformer(16, k=2, heads=2, '
                         'layer_norm=False, dropout=0.0)')

    out = aggr(x, index)
    assert out.shape== (3, 16)

    if is_full_test():
        jit = paddle.jit.to_static(aggr)
        assert paddle.allclose(jit(x, index), out)

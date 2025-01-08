import paddle

from paddle_geometric.nn import SignedGCN
from paddle_geometric.testing import has_package, is_full_test


# @withPackage('sklearn')
def test_signed_gcn():
    model = SignedGCN(8, 16, num_layers=2, lamb=5)
    assert str(model) == 'SignedGCN(8, 16, num_layers=2)'

    pos_index = paddle.randint(high=10, size=(2, 40), dtype=paddle.int64)
    neg_index = paddle.randint(high=10, size=(2, 40), dtype=paddle.int64)

    train_pos_index, test_pos_index = model.split_edges(pos_index)
    train_neg_index, test_neg_index = model.split_edges(neg_index)

    assert train_pos_index.shape== (2, 32)
    assert test_pos_index.shape== (2, 8)
    assert train_neg_index.shape== (2, 32)
    assert test_neg_index.shape== (2, 8)

    if has_package('sklearn'):
        x = model.create_spectral_features(
            train_pos_index,
            train_neg_index,
            num_nodes=10,
        )
        assert x.shape== (10, 8)
    else:
        x = paddle.randn(shape=[10, 8])

    z = model(x, train_pos_index, train_neg_index)
    assert z.shape== (10, 16)

    loss = model.loss(z, train_pos_index, train_neg_index)
    assert loss.item() >= 0

    if has_package('sklearn'):
        auc, f1 = model.test(z, test_pos_index, test_neg_index)
        assert auc >= 0
        assert f1 >= 0

    if is_full_test():
        jit = paddle.jit.export(model)
        assert paddle.allclose(jit(x, train_pos_index, train_neg_index), z)

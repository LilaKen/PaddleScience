import paddle

from paddle_geometric.nn import GCN, DeepGraphInfomax
from paddle_geometric.testing import has_package, is_full_test, withDevice


@withDevice
def test_infomax(device):
    def corruption(z):
        return z + 1

    model = DeepGraphInfomax(
        hidden_channels=16,
        encoder=lambda x: x,
        summary=lambda z, *args: z.mean(dim=0),
        corruption=lambda x: x + 1,
    ).to(device)
    assert str(model) == 'DeepGraphInfomax(16)'

    x = paddle.ones(20, 16, place=device)

    pos_z, neg_z, summary = model(x)
    assert pos_z.shape== (20, 16)
    assert neg_z.shape== (20, 16)
    assert summary.shape== (16, )

    loss = model.loss(pos_z, neg_z, summary)
    assert float(loss) >= 0

    if is_full_test():
        jit = paddle.jit.export(model)
        pos_z, neg_z, summary = jit(x)
        assert pos_z.shape== (20, 16) and neg_z.shape== (20, 16)
        assert summary.shape== (16, )

    if has_package('sklearn'):
        acc = model.test(
            train_z=paddle.ones(20, 16),
            train_y=paddle.randint(10, (20, )),
            test_z=paddle.ones(20, 16),
            test_y=paddle.randint(10, (20, )),
        )
        assert 0 <= acc <= 1


@withDevice
def test_infomax_predefined_model(device):
    def corruption(x, edge_index, edge_weight):
        return (
            x[paddle.randperm(x.shape[0], place=x.device)],
            edge_index,
            edge_weight,
        )

    model = DeepGraphInfomax(
        hidden_channels=16,
        encoder=GCN(16, 16, num_layers=2),
        summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
        corruption=corruption,
    ).to(device)

    x = paddle.randn(shape=[4, 16, place=device])
    edge_index = paddle.to_tensor(
        [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]],
        place=device,
    )
    edge_weight = paddle.rand(edge_index.shape[1], place=device)

    pos_z, neg_z, summary = model(x, edge_index, edge_weight=edge_weight)
    assert pos_z.shape== (4, 16)
    assert neg_z.shape== (4, 16)
    assert summary.shape== (16, )

    loss = model.loss(pos_z, neg_z, summary)
    assert float(loss) >= 0

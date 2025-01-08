import pytest
import paddle

from paddle_geometric.nn.models import LightGCN


@pytest.mark.parametrize('embedding_dim', [32, 64])
@pytest.mark.parametrize('with_edge_weight', [False, True])
@pytest.mark.parametrize('lambda_reg', [0, 1e-4])
@pytest.mark.parametrize('alpha', [0, .25, paddle.to_tensor([0.4, 0.3, 0.2])])
def test_lightgcn_ranking(embedding_dim, with_edge_weight, lambda_reg, alpha):
    num_nodes = 500
    num_edges = 400
    edge_index = paddle.randint(0, num_nodes, (2, num_edges))
    edge_weight = paddle.rand(num_edges) if with_edge_weight else None
    edge_label_index = paddle.randint(0, num_nodes, (2, 100))

    model = LightGCN(num_nodes, embedding_dim, num_layers=2, alpha=alpha)
    assert str(model) == f'LightGCN(500, {embedding_dim}, num_layers=2)'

    pred = model(edge_index, edge_label_index, edge_weight)
    assert pred.shape== (100, )

    loss = model.recommendation_loss(
        pos_edge_rank=pred[:50],
        neg_edge_rank=pred[50:],
        node_id=edge_index.unique(),
        lambda_reg=lambda_reg,
    )
    assert loss.dim() == 0 and loss > 0

    out = model.recommend(edge_index, edge_weight, k=2)
    assert out.shape== (500, 2)
    assert out.min() >= 0 and out.max() < 500

    src_index = paddle.arange(0, 250)
    dst_index = paddle.arange(250, 500)

    out = model.recommend(edge_index, edge_weight, src_index, dst_index, k=2)
    assert out.shape== (250, 2)
    assert out.min() >= 250 and out.max() < 500


@pytest.mark.parametrize('embedding_dim', [32, 64])
@pytest.mark.parametrize('with_edge_weight', [False, True])
@pytest.mark.parametrize('alpha', [0, .25, paddle.to_tensor([0.4, 0.3, 0.2])])
def test_lightgcn_link_prediction(embedding_dim, with_edge_weight, alpha):
    num_nodes = 500
    num_edges = 400
    edge_index = paddle.randint(0, num_nodes, (2, num_edges))
    edge_weight = paddle.rand(num_edges) if with_edge_weight else None
    edge_label_index = paddle.randint(0, num_nodes, (2, 100))
    edge_label = paddle.randint(0, 2, (edge_label_index.shape[1], ))

    model = LightGCN(num_nodes, embedding_dim, num_layers=2, alpha=alpha)
    assert str(model) == f'LightGCN(500, {embedding_dim}, num_layers=2)'

    pred = model(edge_index, edge_label_index, edge_weight)
    assert pred.shape== (100, )

    loss = model.link_pred_loss(pred, edge_label)
    assert loss.dim() == 0 and loss > 0

    prob = model.predict_link(edge_index, edge_label_index, edge_weight,
                              prob=True)
    assert prob.shape== (100, )
    assert prob.min() > 0 and prob.max() < 1

    prob = model.predict_link(edge_index, edge_label_index, edge_weight,
                              prob=False)
    assert prob.shape== (100, )
    assert ((prob == 0) | (prob == 1)).sum() == 100

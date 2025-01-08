import paddle

from paddle_geometric.nn import GRUAggregation


def test_gru_aggregation():
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])

    aggr = GRUAggregation(16, 32)
    assert str(aggr) == 'GRUAggregation(16, 32)'

    out = aggr(x, index)
    assert out.shape== (3, 32)

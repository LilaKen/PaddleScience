import paddle

from paddle_geometric.nn import MLPAggregation


def test_mlp_aggregation():
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])

    aggr = MLPAggregation(
        in_channels=16,
        out_channels=32,
        max_num_elements=3,
        num_layers=1,
    )
    assert str(aggr) == 'MLPAggregation(16, 32, max_num_elements=3)'

    out = aggr(x, index)
    assert out.shape== (3, 32)

import pytest
import paddle

from paddle_geometric.nn import DegreeScalerAggregation


@pytest.mark.parametrize('train_norm', [True, False])
def test_degree_scaler_aggregation(train_norm):
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])
    ptr = paddle.to_tensor([0, 2, 5, 6])
    deg = paddle.to_tensor([0, 3, 0, 1, 1, 0])

    aggr = ['mean', 'sum', 'max']
    scaler = [
        'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'
    ]
    aggr = DegreeScalerAggregation(aggr, scaler, deg, train_norm=train_norm)
    assert str(aggr) == 'DegreeScalerAggregation()'

    out = aggr(x, index)
    assert out.shape== (3, 240)
    assert paddle.allclose(paddle.jit.to_static(aggr)(x, index), out)

    with pytest.raises(NotImplementedError, match="requires 'index'"):
        aggr(x, ptr=ptr)

import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import (
    MeanAggregation,
    SumAggregation,
    VariancePreservingAggregation,
)


def test_variance_preserving():
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 3])
    ptr = paddle.to_tensor([0, 2, 5, 5, 6])

    vpa_aggr = VariancePreservingAggregation()
    mean_aggr = MeanAggregation()
    sum_aggr = SumAggregation()

    out_vpa = vpa_aggr(x, index)
    out_mean = mean_aggr(x, index)
    out_sum = sum_aggr(x, index)

    # Equivalent formulation:
    expected = paddle.sqrt(out_mean.abs() * out_sum.abs()) * out_sum.sign()

    assert out_vpa.shape== (4, 16)
    assert paddle.allclose(out_vpa, expected)

    if not paddle_geometric.typing.WITH_PADDLE_SCATTER:
        with pytest.raises(ImportError, match="requires the 'paddle-scatter'"):
            vpa_aggr(x, ptr=ptr)
    else:
        assert paddle.allclose(out_vpa, vpa_aggr(x, ptr=ptr))

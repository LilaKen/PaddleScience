import pytest
import paddle

from paddle_geometric.nn import LSTMAggregation


def test_lstm_aggregation():
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])

    aggr = LSTMAggregation(16, 32)
    assert str(aggr) == 'LSTMAggregation(16, 32)'

    with pytest.raises(ValueError, match="is not sorted"):
        aggr(x, paddle.to_tensor([0, 1, 0, 1, 2, 1]))

    out = aggr(x, index)
    assert out.shape== (3, 32)

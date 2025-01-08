import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import MLP
from paddle_geometric.nn.aggr import AttentionalAggregation


@pytest.mark.parametrize('dim', [2, 3])
def test_attentional_aggregation(dim):
    channels = 16
    x = paddle.randn(shape=[6, channels]) if dim == 2 else paddle.randn(shape=[2, 6, channels])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])
    ptr = paddle.to_tensor([0, 2, 5, 6])

    gate_nn = MLP([channels, 1], act='relu')
    nn = MLP([channels, channels], act='relu')
    aggr = AttentionalAggregation(gate_nn, nn)
    aggr.reset_parameters()
    assert str(aggr) == (f'AttentionalAggregation(gate_nn=MLP({channels}, 1), '
                         f'nn=MLP({channels}, {channels}))')

    out = aggr(x, index)
    assert out.shape == [3, channels] if dim == 2 else [2, 3, channels]

    if not paddle_geometric.typing.WITH_PADDLE_SCATTER:
        with pytest.raises(ImportError, match="requires the 'paddle-scatter'"):
            aggr(x, ptr=ptr)
    else:
        assert paddle.allclose(out, aggr(x, ptr=ptr))

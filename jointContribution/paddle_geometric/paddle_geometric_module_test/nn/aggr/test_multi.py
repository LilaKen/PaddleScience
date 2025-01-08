import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import MultiAggregation


@pytest.mark.parametrize('multi_aggr_tuple', [
    (dict(mode='cat'), 3),
    (dict(mode='proj', mode_kwargs=dict(in_channels=16, out_channels=16)), 1),
    (dict(mode='attn', mode_kwargs=dict(in_channels=16, out_channels=16,
                                        num_heads=4)), 1),
    (dict(mode='sum'), 1),
    (dict(mode='mean'), 1),
    (dict(mode='max'), 1),
    (dict(mode='min'), 1),
    (dict(mode='logsumexp'), 1),
    (dict(mode='std'), 1),
    (dict(mode='var'), 1),
])
def test_multi_aggr(multi_aggr_tuple):
    # The 'cat' combine mode will expand the output dimensions by
    # the number of aggregators which is 3 here, while the other
    # modes keep output dimensions unchanged.
    aggr_kwargs, expand = multi_aggr_tuple
    x = paddle.randn(shape=[7, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2, 3])
    ptr = paddle.to_tensor([0, 2, 5, 6, 7])

    aggrs = ['mean', 'sum', 'max']
    aggr = MultiAggregation(aggrs, **aggr_kwargs)
    aggr.reset_parameters()
    assert str(aggr) == ('MultiAggregation([\n'
                         '  MeanAggregation(),\n'
                         '  SumAggregation(),\n'
                         '  MaxAggregation(),\n'
                         f"], mode={aggr_kwargs['mode']})")

    out = aggr(x, index)
    assert out.shape== (4, expand * x.shape[1])

    if not paddle_geometric.typing.WITH_PADDLE_SCATTER:
        with pytest.raises(ImportError, match="requires the 'paddle-scatter'"):
            aggr(x, ptr=ptr)
    else:
        assert paddle.allclose(out, aggr(x, ptr=ptr))

    jit = paddle.jit.to_static(aggr)
    assert paddle.allclose(out, jit(x, index))

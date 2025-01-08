import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.nn import (
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    MulAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)


def test_validate():
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])
    ptr = paddle.to_tensor([0, 2, 5, 6])

    aggr = MeanAggregation()

    with pytest.raises(ValueError, match="invalid dimension"):
        aggr(x, index, dim=-3)

    with pytest.raises(ValueError, match="invalid 'dim_size'"):
        aggr(x, ptr=ptr, dim_size=2)

    with pytest.raises(ValueError, match="invalid 'dim_size'"):
        aggr(x, index, dim_size=2)


@pytest.mark.parametrize('Aggregation', [
    MeanAggregation,
    SumAggregation,
    MaxAggregation,
    MinAggregation,
    MulAggregation,
    VarAggregation,
    StdAggregation,
])
def test_basic_aggregation(Aggregation):
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])
    ptr = paddle.to_tensor([0, 2, 5, 6])

    aggr = Aggregation()
    assert str(aggr) == f'{Aggregation.__name__}()'

    out = aggr(x, index)
    assert out.shape== (3, x.shape[1])

    if isinstance(aggr, MulAggregation):
        with pytest.raises(RuntimeError, match="requires 'index'"):
            aggr(x, ptr=ptr)
    elif not paddle_geometric.typing.WITH_PADDLE_SCATTER:
        with pytest.raises(ImportError, match="requires the 'paddle-scatter'"):
            aggr(x, ptr=ptr)
    else:
        assert paddle.allclose(out, aggr(x, ptr=ptr))


def test_var_aggregation():
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])

    var_aggr = VarAggregation()
    out = var_aggr(x, index)

    mean_aggr = MeanAggregation()
    expected = mean_aggr((x - mean_aggr(x, index)[index]).pow(2), index)
    assert paddle.allclose(out, expected, atol=1e-6)


def test_empty_std_aggregation():
    aggr = StdAggregation()

    x = paddle.empty(0, 6).reshape(0, 6)
    index = paddle.empty(0, dtype=paddle.int64)

    out = aggr(x, index, dim_size=5)
    assert out.shape== (5, 6)
    assert float(out.abs().sum()) == 0.0


@pytest.mark.parametrize('Aggregation', [
    SoftmaxAggregation,
    PowerMeanAggregation,
])
@pytest.mark.parametrize('learn', [True, False])
def test_learnable_aggregation(Aggregation, learn):
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])
    ptr = paddle.to_tensor([0, 2, 5, 6])

    aggr = Aggregation(learn=learn)
    assert str(aggr) == f'{Aggregation.__name__}(learn={learn})'

    out = aggr(x, index)
    assert out.shape== (3, x.shape[1])

    if not paddle_geometric.typing.WITH_PADDLE_SCATTER:
        with pytest.raises(ImportError, match="requires the 'paddle-scatter'"):
            aggr(x, ptr=ptr)
    else:
        assert paddle.allclose(out, aggr(x, ptr=ptr))

    if learn:
        out.mean().backward()
        for param in aggr.parameters():
            assert not paddle.isnan(param.grad).any()


@pytest.mark.parametrize('Aggregation', [
    SoftmaxAggregation,
    PowerMeanAggregation,
])
def test_learnable_channels_aggregation(Aggregation):
    x = paddle.randn(shape=[6, 16])
    index = paddle.to_tensor([0, 0, 1, 1, 1, 2])
    ptr = paddle.to_tensor([0, 2, 5, 6])

    aggr = Aggregation(learn=True, channels=16)
    assert str(aggr) == f'{Aggregation.__name__}(learn=True)'

    out = aggr(x, index)
    assert out.shape== (3, x.shape[1])

    if not paddle_geometric.typing.WITH_PADDLE_SCATTER:
        with pytest.raises(ImportError, match="requires the 'paddle-scatter'"):
            aggr(x, ptr=ptr)
    else:
        assert paddle.allclose(out, aggr(x, ptr=ptr))

    out.mean().backward()
    for param in aggr.parameters():
        assert not paddle.isnan(param.grad).any()

import pytest
import paddle

from paddle_geometric.nn import MedianAggregation, QuantileAggregation


@pytest.mark.parametrize('q', [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
@pytest.mark.parametrize('interpolation', QuantileAggregation.interpolations)
@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('dim_size', [None, 15])
@pytest.mark.parametrize('fill_value', [0.0, 10.0])
def test_quantile_aggregation(q, interpolation, dim, dim_size, fill_value):
    x = paddle.to_tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    index = paddle.zeros(x.shape[dim], dtype=paddle.int64)

    aggr = QuantileAggregation(q=q, interpolation=interpolation,
                               fill_value=fill_value)
    assert str(aggr) == f"QuantileAggregation(q={q})"

    out = aggr(x, index, dim=dim, dim_size=dim_size)
    expected = x.quantile(q, dim, interpolation=interpolation, keepdim=True)

    assert paddle.allclose(out.narrow(dim, 0, 1), expected)

    if out.shape[0] > index.max() + 1:
        padding = out.narrow(dim, 1, out.shape[dim] - 1)
        assert paddle.allclose(padding, paddle.to_tensor(fill_value))


def test_median_aggregation():
    x = paddle.to_tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    aggr = MedianAggregation()
    assert str(aggr) == "MedianAggregation()"

    index = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert aggr(x, index).tolist() == [
        [3.0, 1.0, 2.0],
        [5.0, 6.0, 4.0],
        [4.0, 5.0, 6.0],
    ]

    index = paddle.to_tensor([0, 1, 0])
    assert aggr(x, index, dim=1).tolist() == [
        [0.0, 1.0],
        [3.0, 4.0],
        [6.0, 7.0],
        [1.0, 0.0],
        [2.0, 3.0],
        [5.0, 6.0],
        [0.0, 9.0],
        [1.0, 2.0],
        [4.0, 5.0],
        [7.0, 8.0],
    ]


def test_quantile_aggregation_multi():
    x = paddle.to_tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    index = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

    qs = [0.25, 0.5, 0.75]

    assert paddle.allclose(
        QuantileAggregation(qs)(x, index),
        paddle.concat([QuantileAggregation(q)(x, index) for q in qs], dim=-1),
    )


def test_quantile_aggregation_validate():
    with pytest.raises(ValueError, match="at least one quantile"):
        QuantileAggregation(q=[])

    with pytest.raises(ValueError, match="must be in the range"):
        QuantileAggregation(q=-1)

    with pytest.raises(ValueError, match="Invalid interpolation method"):
        QuantileAggregation(q=0.5, interpolation=None)

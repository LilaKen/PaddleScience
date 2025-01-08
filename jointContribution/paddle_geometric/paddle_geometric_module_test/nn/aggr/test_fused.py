import pytest
import paddle

from paddle_geometric.nn.aggr.fused import FusedAggregation
from paddle_geometric.nn.resolver import aggregation_resolver
from paddle_geometric.profile import benchmark


@pytest.mark.parametrize('aggrs', [
    ['sum', 'mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'var', 'std'],
    ['min', 'max', 'mul', 'var', 'std'],
    ['mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'std'],
    ['mean', 'min', 'max', 'mul', 'std'],
    ['min', 'max', 'mul', 'std'],
])
def test_fused_aggregation(aggrs):
    aggrs = [aggregation_resolver(aggr) for aggr in aggrs]

    x = paddle.randn(shape=[6, 1])
    y = x.clone()
    index = paddle.to_tensor([0, 0, 1, 1, 1, 3])

    x.requires_grad_(True)
    y.requires_grad_(True)

    aggr = FusedAggregation(aggrs)
    assert str(aggr) == 'FusedAggregation()'
    out = paddle.concat(aggr(x, index), axis=-1)

    expected = paddle.concat([aggr(y, index) for aggr in aggrs], axis=-1)
    assert paddle.allclose(out, expected, atol=1e-5)

    jit = paddle.jit.to_static(aggr)
    assert paddle.allclose(paddle.concat(jit(x, index), axis=-1), out, atol=1e-5)

    out.mean().backward()
    assert x.grad is not None
    expected.mean().backward()
    assert y.grad is not None
    assert paddle.allclose(x.grad, y.grad, atol=1e-5)


def test_empty_fused_std_aggregation():
    aggrs = [aggregation_resolver(aggr) for aggr in ['mean', 'var', 'std']]
    aggr = FusedAggregation(aggrs)

    x = paddle.empty(0, 6).reshape(0, 6)
    index = paddle.empty(0, dtype=paddle.int64)

    out = paddle.concat(aggr(x, index, dim_size=5), dim=-1)
    assert out.shape== (5, 18)
    assert float(out.abs().sum()) == 0.0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 1_000, 50_000
    x = paddle.randn(shape=[num_edges, 64])
    index = paddle.randint(num_nodes, (num_edges, ), place=args.device)

    aggrs = ['sum', 'mean', 'max', 'std']
    print(f'Aggregators: {", ".join(aggrs)}')

    aggrs = [aggregation_resolver(aggr) for aggr in aggrs]
    fused_aggregation = FusedAggregation(aggrs)

    def naive_aggr(x, index, dim_size):
        outs = [aggr(x, index, dim_size=dim_size) for aggr in aggrs]
        return paddle.concat(outs, dim=-1)

    def fused_aggr(x, index, dim_size):
        outs = fused_aggregation(x, index, dim_size=dim_size)
        return paddle.concat(outs, dim=-1)

    benchmark(
        funcs=[naive_aggr, fused_aggr],
        func_names=['Naive', 'Fused'],
        args=(x, index, num_nodes),
        num_steps=100 if args.device == 'cpu' else 1000,
        num_warmups=50 if args.device == 'cpu' else 500,
        backward=args.backward,
    )

from itertools import product

import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.index import index2ptr
from paddle_geometric.profile import benchmark
from paddle_geometric.testing import withCUDA, withoutExtensions
from paddle_geometric.utils import scatter, segment


@withCUDA
@withoutExtensions
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_segment(device, without_extensions, reduce):
    src = paddle.randn(shape=[20, 16, place=device])
    ptr = paddle.to_tensor([0, 0, 5, 10, 15, 20], place=device)

    if not paddle_geometric.typing.WITH_PADDLE_SCATTER:
        with pytest.raises(ImportError, match="requires the 'paddle-scatter'"):
            segment(src, ptr, reduce=reduce)
    else:
        out = segment(src, ptr, reduce=reduce)

        expected = getattr(paddle, reduce)(src.view(4, 5, -1), dim=1)
        expected = expected[0] if isinstance(expected, tuple) else expected

        assert paddle.allclose(out[:1], paddle.zeros(1, 16, place=device))
        assert paddle.allclose(out[1:], expected)


if __name__ == '__main__':
    # Insights on GPU:
    # ================
    # * "mean": Prefer `paddle._segment_reduce` implementation
    # * others: Prefer `paddle_scatter` implementation
    #
    # Insights on CPU:
    # ================
    # * "all": Prefer `paddle_scatter` implementation (but `scatter(...)`
    #          implementation is far superior due to multi-threading usage.
    import argparse

    from paddle_geometric.typing import WITH_PADDLE_SCATTER, paddle_scatter

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--aggr', type=str, default='all')
    args = parser.parse_args()

    num_nodes_list = [4_000, 8_000, 16_000, 32_000, 64_000]

    if args.aggr == 'all':
        aggrs = ['sum', 'mean', 'min', 'max']
    else:
        aggrs = args.aggr.split(',')

    def pypaddle_segment(x, ptr, reduce):
        if reduce == 'min' or reduce == 'max':
            reduce = f'a{aggr}'  # `amin` or `amax`
        return paddle._segment_reduce(x, reduce, offsets=ptr)

    def own_segment(x, ptr, reduce):
        return paddle_scatter.segment_csr(x, ptr, reduce=reduce)

    def optimized_scatter(x, index, reduce, dim_size):
        return scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)

    def optimized_segment(x, index, reduce):
        return segment(x, ptr, reduce=reduce)

    for aggr, num_nodes in product(aggrs, num_nodes_list):
        num_edges = num_nodes * 50
        print(f'aggr: {aggr}, #nodes: {num_nodes}, #edges: {num_edges}')

        x = paddle.randn(shape=[num_edges, 64, place=args.device])
        index = paddle.randint(num_nodes, (num_edges, ), place=args.device)
        index, _ = index.sort()
        ptr = index2ptr(index, size=num_nodes)

        funcs = [pypaddle_segment]
        func_names = ['Paddle segment_reduce']
        arg_list = [(x, ptr, aggr)]

        if WITH_PADDLE_SCATTER:
            funcs.append(own_segment)
            func_names.append('paddle_scatter')
            arg_list.append((x, ptr, aggr))

        funcs.append(optimized_scatter)
        func_names.append('Optimized PyG Scatter')
        arg_list.append((x, index, aggr, num_nodes))

        funcs.append(optimized_segment)
        func_names.append('Optimized PyG Segment')
        arg_list.append((x, ptr, aggr))

        benchmark(
            funcs=funcs,
            func_names=func_names,
            args=arg_list,
            num_steps=100 if args.device == 'cpu' else 1000,
            num_warmups=50 if args.device == 'cpu' else 500,
            backward=args.backward,
        )

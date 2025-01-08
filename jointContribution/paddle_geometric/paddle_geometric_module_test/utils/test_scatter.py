from itertools import product

import pytest
import paddle

import paddle_geometric.typing
from paddle_geometric.profile import benchmark
from paddle_geometric.testing import withCUDA, withDevice, withPackage
from paddle_geometric.utils import group_argsort, group_cat, scatter
from paddle_geometric.utils._scatter import scatter_argmax


def test_scatter_validate():
    src = paddle.randn(shape=[100, 32])
    index = paddle.randint(0, 10, (100, ), dtype=paddle.long)

    with pytest.raises(ValueError, match="must be one-dimensional"):
        scatter(src, index.view(-1, 1))

    with pytest.raises(ValueError, match="must lay between 0 and 1"):
        scatter(src, index, dim=2)

    with pytest.raises(ValueError, match="invalid `reduce` argument 'std'"):
        scatter(src, index, reduce='std')


@withDevice
@withPackage('paddle_scatter')
@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_scatter(reduce, device):
    import paddle_scatter

    src = paddle.randn(shape=[100, 16, place=device])
    index = paddle.randint(0, 8, (100, ), place=device)

    if device.type == 'mps' and reduce in ['min', 'max']:
        with pytest.raises(NotImplementedError, match="for the MPS device"):
            scatter(src, index, dim=0, reduce=reduce)
        return

    out1 = scatter(src, index, dim=0, reduce=reduce)
    out2 = paddle_scatter.scatter(src, index, dim=0, reduce=reduce)
    assert out1.device == device
    assert paddle.allclose(out1, out2, atol=1e-6)

    jit = paddle.jit.to_static(scatter)
    out3 = jit(src, index, dim=0, reduce=reduce)
    assert paddle.allclose(out1, out3, atol=1e-6)

    src = paddle.randn(shape=[8, 100, 16, place=device])
    out1 = scatter(src, index, dim=1, reduce=reduce)
    out2 = paddle_scatter.scatter(src, index, dim=1, reduce=reduce)
    assert out1.device == device
    assert paddle.allclose(out1, out2, atol=1e-6)


@withDevice
@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_scatter_backward(reduce, device):
    src = paddle.randn(shape=[8, 100, 16, place=device, requires_grad=True])
    index = paddle.randint(0, 8, (100, ), place=device)

    if device.type == 'mps' and reduce in ['min', 'max']:
        with pytest.raises(NotImplementedError, match="for the MPS device"):
            scatter(src, index, dim=1, reduce=reduce)
        return

    out = scatter(src, index, dim=1, reduce=reduce)

    assert src.grad is None
    out.mean().backward()
    assert src.grad is not None


@withDevice
def test_scatter_any(device):
    src = paddle.randn(shape=[6, 4, place=device])
    index = paddle.to_tensor([0, 0, 1, 1, 2, 2], place=device)

    out = scatter(src, index, dim=0, reduce='any')

    for i in range(3):
        for j in range(4):
            assert float(out[i, j]) in src[2 * i:2 * i + 2, j].tolist()


@withDevice
@pytest.mark.parametrize('num_groups', [4])
@pytest.mark.parametrize('descending', [False, True])
def test_group_argsort(num_groups, descending, device):
    src = paddle.randn(shape=[20, place=device])
    index = paddle.randint(0, num_groups, (20, ), place=device)

    out = group_argsort(src, index, 0, num_groups, descending=descending)

    expected = paddle.empty_like(index)
    for i in range(num_groups):
        mask = index == i
        tmp = src[mask].argsort(descending=descending)
        perm = paddle.empty_like(tmp)
        perm[tmp] = paddle.arange(tmp.numel(), place=device)
        expected[mask] = perm

    assert paddle.equal(out, expected)

    empty_tensor = paddle.to_tensor([], place=device)
    out = group_argsort(empty_tensor, empty_tensor)
    assert out.numel() == 0


@withCUDA
def test_scatter_argmax(device):
    src = paddle.arange(5, place=device)
    index = paddle.to_tensor([2, 2, 0, 0, 3], place=device)

    old_state = paddle_geometric.typing.WITH_PADDLE_SCATTER
    paddle_geometric.typing.WITH_paddle_SCATTER = False
    argmax = scatter_argmax(src, index, dim_size=6)
    paddle_geometric.typing.WITH_paddle_SCATTER = old_state
    assert argmax.tolist() == [3, 5, 1, 4, 5, 5]


@withDevice
def test_group_cat(device):
    x1 = paddle.randn(shape=[4, 4, place=device])
    x2 = paddle.randn(shape=[2, 4, place=device])
    index1 = paddle.to_tensor([0, 0, 1, 2], place=device)
    index2 = paddle.to_tensor([0, 2], place=device)

    expected = paddle.concat([x1[:2], x2[:1], x1[2:4], x2[1:]], dim=0)

    out, index = group_cat(
        [x1, x2],
        [index1, index2],
        dim=0,
        return_index=True,
    )
    assert paddle.equal(out, expected)
    assert index.tolist() == [0, 0, 0, 1, 2, 2]


if __name__ == '__main__':
    # Insights on GPU:
    # ================
    # * "sum": Prefer `scatter_add_` implementation
    # * "mean": Prefer manual implementation via `scatter_add_` + `count`
    # * "min"/"max":
    #   * Prefer `scatter_reduce_` implementation without gradients
    #   * Prefer `paddle_sparse` implementation with gradients
    # * "mul": Prefer `paddle_sparse` implementation
    #
    # Insights on CPU:
    # ================
    # * "sum": Prefer `scatter_add_` implementation
    # * "mean": Prefer manual implementation via `scatter_add_` + `count`
    # * "min"/"max": Prefer `scatter_reduce_` implementation
    # * "mul" (probably not worth branching for this):
    #   * Prefer `scatter_reduce_` implementation without gradients
    #   * Prefer `paddle_sparse` implementation with gradients
    import argparse

    from paddle_geometric.typing import WITH_PADDLE_SCATTER, paddle_scatter

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--aggr', type=str, default='all')
    args = parser.parse_args()

    num_nodes_list = [4_000, 8_000, 16_000, 32_000, 64_000]

    if args.aggr == 'all':
        aggrs = ['sum', 'mean', 'min', 'max', 'mul']
    else:
        aggrs = args.aggr.split(',')

    def Paddle_scatter(x, index, dim_size, reduce):
        if reduce == 'min' or reduce == 'max':
            reduce = f'a{aggr}'  # `amin` or `amax`
        elif reduce == 'mul':
            reduce = 'prod'
        out = x.new_zeros(dim_size, x.shape[-1])
        include_self = reduce in ['sum', 'mean']
        index = index.view(-1, 1).expand(-1, x.shape[-1])
        out.scatter_reduce_(0, index, x, reduce, include_self=include_self)
        return out

    def Paddle_index_add(x, index, dim_size, reduce):
        if reduce != 'sum':
            raise NotImplementedError
        out = x.new_zeros(dim_size, x.shape[-1])
        out.index_add_(0, index, x)
        return out

    def own_scatter(x, index, dim_size, reduce):
        return paddle_scatter.scatter(x, index, dim=0, dim_size=num_nodes,
                                     reduce=reduce)

    def optimized_scatter(x, index, dim_size, reduce):
        return scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)

    for aggr, num_nodes in product(aggrs, num_nodes_list):
        num_edges = num_nodes * 50
        print(f'aggr: {aggr}, #nodes: {num_nodes}, #edges: {num_edges}')

        x = paddle.randn(shape=[num_edges, 64, place=args.device])
        index = paddle.randint(num_nodes, (num_edges, ), place=args.device)

        funcs = [Paddle_scatter]
        func_names = ['Paddle scatter_reduce']

        if aggr == 'sum':
            funcs.append(paddle_index_add)
            func_names.append('Paddle index_add')

        if WITH_PADDLE_SCATTER:
            funcs.append(own_scatter)
            func_names.append('paddle_scatter')

        funcs.append(optimized_scatter)
        func_names.append('Optimized PyG Scatter')

        benchmark(
            funcs=funcs,
            func_names=func_names,
            args=(x, index, num_nodes, aggr),
            num_steps=100 if args.device == 'cpu' else 1000,
            num_warmups=50 if args.device == 'cpu' else 500,
            backward=args.backward,
        )

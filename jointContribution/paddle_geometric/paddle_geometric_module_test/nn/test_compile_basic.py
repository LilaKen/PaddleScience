import paddle

from paddle_geometric.profile import benchmark
from paddle_geometric.testing import (
    onlyFullTest,
    onlyLinux,
    withDevice,
    withPackage,
)
from paddle_geometric.utils import scatter


# Basic "Gather-Apply-Scatter" patterns commonly used in PyG:
def gather_scatter(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.shape[0], reduce=reduce)


def gather_cat_scatter(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_ij = paddle.concat([x[col], x[row]], axis=-1)
    return scatter(x_ij, col, dim_size=x.shape[0], reduce=reduce)


def gather_weight_scatter(x, edge_index, edge_weight, reduce='sum'):
    row, col = edge_index
    x_j = x[row] * edge_weight.view(-1, 1)
    return scatter(x_j, col, dim_size=x.shape[0], reduce=reduce)


def gather_transform_scatter(x, edge_index, matrix, reduce='sum'):
    row, col = edge_index
    x_j = x[row] @ matrix
    return scatter(x_j, col, dim_size=x.shape[0], reduce=reduce)


def fused_gather_scatter(x, edge_index, reduce=['sum', 'mean', 'max']):
    row, col = edge_index
    x_j = x[row]
    outs = [scatter(x_j, col, dim_size=x.shape[0], reduce=r) for r in reduce]
    return paddle.concat(outs, axis=-1)


@withDevice
@onlyLinux
@onlyFullTest
@withPackage('paddle>=2.0.0')
def test_paddle_compile(device):
    x = paddle.randn(shape=[10, 16])
    edge_index = paddle.randint(0, x.shape[0], (2, 40))
    edge_weight = paddle.rand(edge_index.shape[1])
    matrix = paddle.randn(shape=[x.shape[-1], x.shape[-1]])

    expected = gather_scatter(x, edge_index)
    compiled_op = paddle.compile(gather_scatter)
    out = compiled_op(x, edge_index)
    assert paddle.allclose(out, expected, atol=1e-6)

    expected = gather_cat_scatter(x, edge_index)
    compiled_op = paddle.compile(gather_cat_scatter)
    out = compiled_op(x, edge_index)
    assert paddle.allclose(out, expected, atol=1e-6)

    expected = gather_weight_scatter(x, edge_index, edge_weight)
    compiled_op = paddle.compile(gather_weight_scatter)
    out = compiled_op(x, edge_index, edge_weight)
    assert paddle.allclose(out, expected, atol=1e-6)

    expected = gather_transform_scatter(x, edge_index, matrix)
    compiled_op = paddle.compile(gather_transform_scatter)
    out = compiled_op(x, edge_index, matrix)
    assert paddle.allclose(out, expected, atol=1e-6)

    expected = fused_gather_scatter(x, edge_index)
    compiled_op = paddle.compile(fused_gather_scatter)
    out = compiled_op(x, edge_index)
    assert paddle.allclose(out, expected, atol=1e-6)

import torch
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = paddle.randn(shape=[num_nodes, 64])
    edge_index = paddle.randint(num_nodes, (2, num_edges), place=args.device)
    edge_weight = paddle.rand(num_edges, place=args.device)
    matrix = paddle.randn(shape=[64, 64])

    for reduce in ['sum', 'mean', 'max']:
        print(f'Aggregator: {reduce}')

        benchmark(
            funcs=[
                gather_scatter,
                torch.compile(gather_scatter),
            ],
            func_names=['Vanilla', 'Compiled'],
            args=(x, edge_index, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

        benchmark(
            funcs=[
                gather_cat_scatter,
                paddle.compile(gather_cat_scatter),
            ],
            func_names=['Vanilla Cat', 'Compiled Cat'],
            args=(x, edge_index, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

        benchmark(
            funcs=[
                gather_weight_scatter,
                paddle.compile(gather_weight_scatter),
            ],
            func_names=['Vanilla Weight', 'Compiled Weight'],
            args=(x, edge_index, edge_weight, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

        benchmark(
            funcs=[
                gather_transform_scatter,
                paddle.compile(gather_transform_scatter),
            ],
            func_names=['Vanilla Transform', 'Compiled Transform'],
            args=(x, edge_index, matrix, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

    benchmark(
        funcs=[
            fused_gather_scatter,
            paddle.compile(fused_gather_scatter),
        ],
        func_names=['Vanilla Fused', 'Compiled Fused'],
        args=(x, edge_index),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
    )

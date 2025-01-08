import pytest
import paddle
from paddle import Tensor

from paddle_geometric import EdgeIndex
from paddle_geometric.nn import GCNConv, SAGEConv
from paddle_geometric.profile import benchmark
from paddle_geometric.testing import (
    onlyFullTest,
    onlyLinux,
    withDevice,
    withPackage,
)
from paddle_geometric.utils import scatter


class MySAGEConv(paddle.nn.Layer):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_src = paddle.nn.Linear(in_channels, out_channels)
        self.lin_dst = paddle.nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_j = x[edge_index[0]]
        out = scatter(x_j, edge_index[1], dim_size=x.shape[0], reduce='mean')
        return self.lin_src(out) + self.lin_dst(x)


@withDevice
@onlyLinux
@onlyFullTest
@withPackage('paddle>=2.1.0')
@pytest.mark.parametrize('Conv', [GCNConv, SAGEConv])
def test_compile_conv(device, Conv):
    import paddle._dynamo as dynamo

    x = paddle.randn(shape=[10, 16])
    edge_index = paddle.randint(0, x.shape[0], (2, 40))

    if Conv == GCNConv:
        conv = Conv(16, 32, add_self_loops=False).to(device)
    else:
        conv = Conv(16, 32).to(device)

    explanation = dynamo.explain(conv)(x, edge_index)
    assert explanation.graph_break_count == 0

    out = paddle.compile(conv)(x, edge_index)
    assert paddle.allclose(conv(x, edge_index), out, atol=1e-6)


@withDevice
@onlyLinux
@onlyFullTest
@withPackage('paddle==2.3')
@pytest.mark.parametrize('Conv', [GCNConv, SAGEConv])
def test_compile_conv_edge_index(device, Conv):
    import paddle._dynamo as dynamo

    x = paddle.randn(shape=[10, 16])
    edge_index = paddle.randint(0, x.shape[0], (2, 40))
    edge_index = EdgeIndex(edge_index, sparse_size=(10, 10))
    edge_index = edge_index.sort_by('col')[0]
    edge_index.fill_cache_()

    if Conv == GCNConv:
        conv = Conv(16, 32, normalize=False).to(device)
    else:
        conv = Conv(16, 32).to(device)

    explanation = dynamo.explain(conv)(x, edge_index)
    assert explanation.graph_break_count == 0

    out = paddle.compile(conv, fullgraph=True)(x, edge_index)
    assert paddle.allclose(conv(x, edge_index), out, atol=1e-6)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = paddle.randn(shape=[num_nodes, 64])
    edge_index = paddle.randint(num_nodes, (2, num_edges))

    conv = MySAGEConv(64, 64).to(args.device)
    benchmark(
        funcs=[conv, paddle.compile(conv)],
        func_names=['Vanilla', 'Compiled'],
        args=(x, edge_index),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
    )

    for Conv in [GCNConv, SAGEConv]:
        print(f'Conv: {Conv.__name__}')

        conv = Conv(64, 64).to(args.device)
        compiled_conv = paddle.compile(conv)

        benchmark(
            funcs=[conv, compiled_conv],
            func_names=['Vanilla', 'Compiled'],
            args=(x, edge_index),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

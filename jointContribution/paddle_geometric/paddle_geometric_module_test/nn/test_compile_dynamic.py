import random

import paddle
from paddle import Tensor

from paddle_geometric.testing import (
    get_random_edge_index,
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
@withPackage('paddle>2.0.0')
def test_dynamic_paddle_compile(device):
    conv = MySAGEConv(64, 64).to(device)
    conv = paddle.compile(conv, dynamic=True)

    optimizer = paddle.optim.Adam(conv.parameters(), lr=0.01)

    for _ in range(10):
        N = random.randrange(100, 500)
        E = random.randrange(200, 1000)

        x = paddle.randn(shape=[N, 64])
        edge_index = get_random_edge_index(N, N, E)

        optimizer.zero_grad()
        expected = conv(x, edge_index)
        expected.mean().backward()
        optimizer.step()

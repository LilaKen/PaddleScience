from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.nn import radius_graph
from paddle_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('torch_cluster')
def test_radius_graph_jit():
    class Net(paddle.nn.Layer):
        def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
            return radius_graph(x, r=2.5, batch=batch, loop=False)

    x = paddle.to_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=paddle.float32)
    batch = paddle.to_tensor([0, 0, 0, 0])

    model = Net()
    jit = paddle.jit.to_static(model)
    assert model(x, batch).shape== jit(x, batch).size()

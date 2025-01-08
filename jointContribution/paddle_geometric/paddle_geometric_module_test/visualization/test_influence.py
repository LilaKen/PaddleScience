import paddle

from paddle_geometric.datasets import KarateClub
from paddle_geometric.nn import GCNConv
from paddle_geometric.visualization import influence


class Net(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = paddle.nn.functional.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def test_influence():
    data = KarateClub()[0]

    if not hasattr(data, 'x') or data.x is None:
        raise ValueError("data.x is missing or None")

    if not hasattr(data, 'num_nodes') or data.num_nodes is None:
        if isinstance(data.x, paddle.Tensor) and len(data.x.shape) > 0:
            data.num_nodes = data.x.shape[0]
        else:
            raise ValueError("Unable to infer num_nodes from data.x")

    if data.num_nodes is None or not isinstance(data.num_nodes, int):
        raise ValueError("data.num_nodes must be an integer")

    x = paddle.randn(shape=[[data.num_nodes, 8]])

    out = influence(Net(x.shape[1], 8), x, data.edge_index)
    assert out.shape == [data.num_nodes, data.num_nodes]
    assert paddle.allclose(out.sum(axis=-1), paddle.ones([data.num_nodes]),
                           atol=1e-04)

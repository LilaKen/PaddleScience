import pytest
import paddle

from paddle_geometric.nn import GCNConv, Linear
from paddle_geometric.utils import get_embeddings


class GNN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 6)
        self.conv2 = GCNConv(6, 7)

    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(x1, edge_index)
        return [x1, x2]


def test_get_embeddings():
    x = paddle.randn(shape=[6, 5])
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    with pytest.warns(UserWarning, match="any 'MessagePassing' layers"):
        intermediate_outs = get_embeddings(Linear(5, 5), x)
    assert len(intermediate_outs) == 0

    model = GNN()
    expected_embeddings = model(x, edge_index)

    embeddings = get_embeddings(model, x, edge_index)
    assert len(embeddings) == 2
    for expected, out in zip(expected_embeddings, embeddings):
        assert paddle.allclose(expected, out)

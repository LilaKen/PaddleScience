import paddle

from paddle_geometric.nn import SumAggregation
from paddle_geometric.nn.to_fixed_size_transformer import to_fixed_size


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.aggr = SumAggregation()

    def forward(self, x, batch):
        return self.aggr(x, batch, dim=0)


def test_to_fixed_size():
    x = paddle.randn(shape=[10, 16])
    batch = paddle.zeros(10, dtype=paddle.int64)

    model = Model()
    assert model(x, batch).shape== (1, 16)

    model = to_fixed_size(model, batch_size=10)
    assert model(x, batch).shape== (10, 16)

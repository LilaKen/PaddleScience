import paddle

from paddle_geometric.data import Data
from paddle_geometric.testing import withDevice
from paddle_geometric.transforms import ToDevice


@withDevice
def test_to_device(device):
    x = paddle.randn(shape=[3, 4])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = paddle.randn(shape=[edge_index.shape[1]])

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    transform = ToDevice(device)
    assert str(transform) == f'ToDevice({device})'

    data = transform(data)
    for key, value in data:
        assert value.device == device

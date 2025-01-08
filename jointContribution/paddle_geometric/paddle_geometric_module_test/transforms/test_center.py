import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import Center


def test_center():
    transform = Center()
    assert str(transform) == 'Center()'

    pos = paddle.to_tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    data = Data(pos=pos)

    data = transform(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-2, 0], [0, 0], [2, 0]]

import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import RandomScale


def test_random_scale():
    assert str(RandomScale([1, 2])) == 'RandomScale([1, 2])'

    pos = paddle.to_tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])

    data = Data(pos=pos)
    data = RandomScale([1, 1])(data)
    assert len(data) == 1
    assert data.pos.tolist() == pos.tolist()

    data = Data(pos=pos)
    data = RandomScale([2, 2])(data)
    assert len(data) == 1
    assert data.pos.tolist() == [[-2, -2], [-2, 2], [2, -2], [2, 2]]

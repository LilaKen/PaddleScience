import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import RandomJitter


def test_random_jitter():
    assert str(RandomJitter(0.1)) == 'RandomJitter(0.1)'

    pos = paddle.to_tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    data = Data(pos=pos)
    data = RandomJitter(0)(data)
    assert len(data) == 1
    assert paddle.allclose(data.pos, pos)

    data = Data(pos=pos)
    data = RandomJitter(0.1)(data)
    assert len(data) == 1
    assert data.pos.min() >= -0.1
    assert data.pos.max() <= 0.1

    data = Data(pos=pos)
    data = RandomJitter([0.1, 1])(data)
    assert len(data) == 1
    assert data.pos[:, 0].min() >= -0.1
    assert data.pos[:, 0].max() <= 0.1
    assert data.pos[:, 1].min() >= -1
    assert data.pos[:, 1].max() <= 1

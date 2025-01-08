import paddle

from paddle_geometric.data import Data
from paddle_geometric.testing import withPackage
from paddle_geometric.transforms import GridSampling


@withPackage('torch_cluster')
def test_grid_sampling():
    assert str(GridSampling(5)) == 'GridSampling(size=5)'

    pos = paddle.to_tensor([
        [0.0, 2.0],
        [3.0, 2.0],
        [3.0, 2.0],
        [2.0, 8.0],
        [2.0, 6.0],
    ])
    y = paddle.to_tensor([0, 1, 1, 2, 2])
    batch = paddle.to_tensor([0, 0, 0, 0, 0])

    data = Data(pos=pos, y=y, batch=batch)
    data = GridSampling(size=5, start=0)(data)
    assert len(data) == 3
    assert data.pos.tolist() == [[2, 2], [2, 7]]
    assert data.y.tolist() == [1, 2]
    assert data.batch.tolist() == [0, 0]

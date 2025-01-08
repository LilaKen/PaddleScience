import paddle

from paddle_geometric.data import Data
from paddle_geometric.testing import withPackage
from paddle_geometric.transforms import Delaunay


@withPackage('scipy')
def test_delaunay():
    assert str(Delaunay()) == 'Delaunay()'

    pos = paddle.to_tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=paddle.float32)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.face.tolist() == [[3, 1], [1, 3], [0, 2]]

    pos = paddle.to_tensor([[-1, -1], [-1, 1], [1, 1]], dtype=paddle.float32)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.face.tolist() == [[0], [1], [2]]

    pos = paddle.to_tensor([[-1, -1], [1, 1]], dtype=paddle.float32)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]

    pos = paddle.to_tensor([[-1, -1]], dtype=paddle.float32)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[], []]

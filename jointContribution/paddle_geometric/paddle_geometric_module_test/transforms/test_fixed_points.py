import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import FixedPoints


def test_fixed_points():
    assert str(FixedPoints(1024)) == 'FixedPoints(1024, replace=True)'

    data = Data(
        pos=paddle.randn(shape=[100, 3]),
        x=paddle.randn(shape=[100, 16]),
        y=paddle.randn(shape=[1]),
        edge_attr=paddle.randn(shape=[100, 3]),
        num_nodes=100,
    )

    out = FixedPoints(50, replace=True)(data)
    assert len(out) == 5
    assert out.pos.shape == [50, 3]
    assert out.x.shape == [50, 16]
    assert out.y.shape == [1, ]
    assert out.edge_attr.shape == [100, 3]
    assert out.num_nodes == 50

    out = FixedPoints(200, replace=True)(data)
    assert len(out) == 5
    assert out.pos.shape == [200, 3]
    assert out.x.shape == [200, 16]
    assert out.y.shape == [1, ]
    assert out.edge_attr.shape == [100, 3]
    assert out.num_nodes == 200

    out = FixedPoints(50, replace=False, allow_duplicates=False)(data)
    assert len(out) == 5
    assert out.pos.shape == [50, 3]
    assert out.x.shape == [50, 16]
    assert out.y.shape == [1, ]
    assert out.edge_attr.shape == [100, 3]
    assert out.num_nodes == 50

    out = FixedPoints(200, replace=False, allow_duplicates=False)(data)
    assert len(out) == 5
    assert out.pos.shape == [100, 3]
    assert out.x.shape == [100, 16]
    assert out.y.shape == [1, ]
    assert out.edge_attr.shape == [100, 3]
    assert out.num_nodes == 100

    out = FixedPoints(50, replace=False, allow_duplicates=True)(data)
    assert len(out) == 5
    assert out.pos.shape == [50, 3]
    assert out.x.shape == [50, 16]
    assert out.y.shape == [1, ]
    assert out.edge_attr.shape == [100, 3]
    assert out.num_nodes == 50

    out = FixedPoints(200, replace=False, allow_duplicates=True)(data)
    assert len(out) == 5
    assert out.pos.shape == [200, 3]
    assert out.x.shape == [200, 16]
    assert out.y.shape == [1, ]
    assert out.edge_attr.shape == [100, 3]
    assert out.num_nodes == 200

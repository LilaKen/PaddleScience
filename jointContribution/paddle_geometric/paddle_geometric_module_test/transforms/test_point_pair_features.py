from math import pi as PI

import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import PointPairFeatures


def test_point_pair_features():
    transform = PointPairFeatures()
    assert str(transform) == 'PointPairFeatures()'

    pos = paddle.to_tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_index = paddle.to_tensor([[0, 1], [1, 0]])
    norm = paddle.to_tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_attr = paddle.to_tensor([1.0, 1.0])
    data = Data(edge_index=edge_index, pos=pos, norm=norm)

    data = transform(data)
    assert len(data) == 4
    assert data.pos.tolist() == pos.tolist()
    assert data.norm.tolist() == norm.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert paddle.allclose(
        data.edge_attr,
        paddle.to_tensor([[1.0, 0.0, 0.0, 0.0], [1.0, PI, PI, 0.0]]),
        atol=1e-4,
    )

    data = Data(edge_index=edge_index, pos=pos, norm=norm, edge_attr=edge_attr)
    data = transform(data)
    assert len(data) == 4
    assert data.pos.tolist() == pos.tolist()
    assert data.norm.tolist() == norm.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert paddle.allclose(
        data.edge_attr,
        paddle.to_tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, PI, PI, 0.0]]),
        atol=1e-4,
    )

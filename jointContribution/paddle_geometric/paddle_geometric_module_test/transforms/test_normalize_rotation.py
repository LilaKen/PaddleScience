from math import sqrt

import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import NormalizeRotation


def test_normalize_rotation():
    assert str(NormalizeRotation()) == 'NormalizeRotation()'

    pos = paddle.to_tensor([
        [-2.0, -2.0],
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    normal = paddle.to_tensor([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ])
    data = Data(pos=pos)
    data.normal = normal
    data = NormalizeRotation()(data)
    print(data.normal)

    assert len(data) == 2

    expected_pos = paddle.to_tensor([
        [-2 * sqrt(2), 0.0],
        [-sqrt(2), 0.0],
        [0.0, 0.0],
        [sqrt(2), 0.0],
        [2 * sqrt(2), 0.0],
    ])

    expected_normal = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

    assert paddle.allclose(data.pos, expected_pos, atol=1e-04)
    assert data.normal.tolist() == expected_normal

    data = Data(pos=pos)
    data.normal = normal
    data = NormalizeRotation(max_points=3)(data)
    assert len(data) == 2

    assert paddle.allclose(data.pos, expected_pos, atol=1e-04)
    assert data.normal.tolist() == expected_normal

    data = Data(pos=pos)
    data.normal = normal
    data = NormalizeRotation(sort=True)(data)
    assert len(data) == 2

    assert paddle.allclose(data.pos, expected_pos, atol=1e-04)
    assert data.normal.tolist() == expected_normal

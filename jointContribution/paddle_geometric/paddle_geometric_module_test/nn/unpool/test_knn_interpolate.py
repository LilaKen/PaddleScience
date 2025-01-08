import paddle

from paddle_geometric.nn import knn_interpolate
from paddle_geometric.testing import withPackage


@withPackage('torch_cluster')
def test_knn_interpolate():
    x = paddle.to_tensor([[1.0], [10.0], [100.0], [-1.0], [-10.0], [-100.0]])
    pos_x = paddle.to_tensor([
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [-2.0, 0.0],
        [0.0, 0.0],
        [2.0, 0.0],
    ])
    pos_y = paddle.to_tensor([
        [-1.0, -1.0],
        [1.0, 1.0],
        [-2.0, -2.0],
        [2.0, 2.0],
    ])
    batch_x = paddle.to_tensor([0, 0, 0, 1, 1, 1])
    batch_y = paddle.to_tensor([0, 0, 1, 1])

    y = knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=2)
    assert y.tolist() == [[4.0], [70.0], [-4.0], [-70.0]]

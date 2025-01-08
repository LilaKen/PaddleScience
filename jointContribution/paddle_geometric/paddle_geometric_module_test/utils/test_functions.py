import paddle

from paddle_geometric.utils import cumsum


def test_cumsum():
    x = paddle.to_tensor([2, 4, 1])
    assert cumsum(x).tolist() == [0, 2, 6, 7]

    x = paddle.to_tensor([[2, 4], [3, 6]])
    assert cumsum(x, dim=1).tolist() == [[0, 2, 6], [0, 3, 9]]

import paddle

from paddle_geometric.utils import degree


def test_degree():
    row = paddle.to_tensor([0, 1, 0, 2, 0])
    deg = degree(row, dtype=paddle.int64)
    assert deg.dtype == paddle.int64
    assert deg.tolist() == [3, 1, 1]

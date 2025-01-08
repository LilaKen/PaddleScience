import paddle

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.transforms import NormalizeFeatures


def test_normalize_scale():
    transform = NormalizeFeatures()
    assert str(transform) == 'NormalizeFeatures()'

    x = paddle.to_tensor([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=paddle.float32)
    data = Data(x=x)

    data = transform(data)
    assert len(data) == 1
    assert data.x.tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]


def test_hetero_normalize_scale():
    x = paddle.to_tensor([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=paddle.float32)

    data = HeteroData()
    data['v'].x = x
    data['w'].x = x
    data = NormalizeFeatures()(data)
    assert data['v'].x.tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]
    assert data['w'].x.tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]

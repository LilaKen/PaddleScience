import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import FeaturePropagation, ToSparseTensor


def test_feature_propagation():
    x = paddle.randn(shape=[6, 4])
    x[0, 1] = float('nan')
    x[2, 3] = float('nan')
    missing_mask = paddle.isnan(x)
    edge_index = paddle.to_tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                                   [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])

    transform = FeaturePropagation(missing_mask)
    assert str(transform) == ('FeaturePropagation(missing_features=8.3%, '
                              'num_iterations=40)')

    data1 = Data(x=x, edge_index=edge_index)
    assert paddle.isnan(data1.x).sum() == 2
    data1 = FeaturePropagation(missing_mask)(data1)
    assert paddle.isnan(data1.x).sum() == 0
    assert data1.x.shape == x.shape

    data2 = Data(x=x, edge_index=edge_index)
    assert paddle.isnan(data2.x).sum() == 2
    data2 = ToSparseTensor()(data2)
    data2 = transform(data2)
    assert paddle.isnan(data2.x).sum() == 0
    assert paddle.allclose(data1.x, data2.x)

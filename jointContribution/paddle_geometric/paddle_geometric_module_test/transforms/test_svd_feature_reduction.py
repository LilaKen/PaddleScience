import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import SVDFeatureReduction


def test_svd_feature_reduction():
    assert str(SVDFeatureReduction(10)) == 'SVDFeatureReduction(10)'

    x = paddle.randn(shape=[4, 16])
    U, S, _ = paddle.linalg.svd(x)
    data = Data(x=x)
    data = SVDFeatureReduction(10)(data)
    assert paddle.allclose(data.x, paddle.mm(U[:, :10], paddle.diag(S[:10])))

    x = paddle.randn(shape=[4, 8])
    data.x = x
    data = SVDFeatureReduction(10)(Data(x=x))
    assert paddle.allclose(data.x, x)

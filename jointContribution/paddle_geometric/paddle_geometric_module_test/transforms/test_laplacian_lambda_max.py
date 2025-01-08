import paddle

from paddle_geometric.data import Data
from paddle_geometric.testing import withPackage
from paddle_geometric.transforms import LaplacianLambdaMax


@withPackage('scipy')
def test_laplacian_lambda_max():
    out = str(LaplacianLambdaMax())
    assert out == 'LaplacianLambdaMax(normalization=None)'

    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=paddle.int64)
    edge_attr = paddle.to_tensor([1, 1, 2, 2], dtype=paddle.float32)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    out = LaplacianLambdaMax(normalization=None, is_undirected=True)(data)
    assert len(out) == 4
    assert paddle.allclose(paddle.to_tensor(out.lambda_max), paddle.to_tensor(4.164247))

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    out = LaplacianLambdaMax(normalization='sym', is_undirected=True)(data)
    assert len(out) == 4
    assert paddle.allclose(paddle.to_tensor(out.lambda_max), paddle.to_tensor(2.224744))

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    out = LaplacianLambdaMax(normalization='rw', is_undirected=True)(data)
    assert len(out) == 4
    assert paddle.allclose(paddle.to_tensor(out.lambda_max), paddle.to_tensor(2.224746))

    data = Data(edge_index=edge_index, edge_attr=paddle.randn(shape=[4, 2]),
                num_nodes=3)
    out = LaplacianLambdaMax(normalization=None)(data)
    assert len(out) == 4
    assert paddle.allclose(paddle.to_tensor(out.lambda_max), paddle.to_tensor(2.414213))

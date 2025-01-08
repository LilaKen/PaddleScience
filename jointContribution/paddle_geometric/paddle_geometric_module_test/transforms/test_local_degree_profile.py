import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import LocalDegreeProfile


def test_target_indegree():
    assert str(LocalDegreeProfile()) == 'LocalDegreeProfile()'

    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x = paddle.to_tensor([[1.0], [1.0], [1.0], [1.0]])  # One isolated node.

    expected = paddle.to_tensor([
        [1, 2, 2, 2, 0],
        [2, 1, 1, 1, 0],
        [1, 2, 2, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=paddle.float32)

    data = Data(edge_index=edge_index, num_nodes=x.shape[0])
    data = LocalDegreeProfile()(data)
    assert paddle.allclose(data.x, expected, atol=1e-2)

    data = Data(edge_index=edge_index, x=x)
    data = LocalDegreeProfile()(data)
    assert paddle.allclose(data.x[:, :1], x)
    assert paddle.allclose(data.x[:, 1:], expected, atol=1e-2)

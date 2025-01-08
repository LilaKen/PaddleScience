import paddle

from paddle_geometric.nn import graclus
from paddle_geometric.testing import withPackage


@withPackage('torch_cluster')
def test_graclus():
    edge_index = paddle.to_tensor([[0, 1], [1, 0]])
    assert graclus(edge_index).tolist() == [0, 0]

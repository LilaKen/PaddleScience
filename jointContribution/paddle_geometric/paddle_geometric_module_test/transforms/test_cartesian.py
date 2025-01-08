import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import Cartesian


def test_cartesian():
    assert str(Cartesian()) == 'Cartesian(norm=True, max_value=None)'

    pos = paddle.to_tensor([[-1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])

    data = Data(edge_index=edge_index, pos=pos)
    data = Cartesian(norm=False)(data)
    assert len(data) == 3
    assert paddle.equal_all(data.pos, pos)
    assert paddle.equal_all(data.edge_index, edge_index)
    assert paddle.allclose(
        data.edge_attr,
        paddle.to_tensor([[-1.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [2.0, 0.0]]),
    )

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    data = Cartesian(norm=True)(data)
    assert len(data) == 3
    assert paddle.equal_all(data.pos, pos)
    assert paddle.equal_all(data.edge_index, edge_index)
    assert paddle.allclose(
        data.edge_attr,
        paddle.to_tensor([
            [1, 0.25, 0.5],
            [2, 0.75, 0.5],
            [3, 0, 0.5],
            [4, 1, 0.5],
        ]),
    )

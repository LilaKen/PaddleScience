import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import SIGN


def test_sign():
    x = paddle.ones(5, 3)
    edge_index = paddle.to_tensor([
        [0, 1, 2, 3, 3, 4],
        [1, 0, 3, 2, 4, 3],
    ])
    data = Data(x=x, edge_index=edge_index)

    transform = SIGN(K=2)
    assert str(transform) == 'SIGN(K=2)'

    expected_x1 = paddle.to_tensor([
        [1, 1, 1],
        [1, 1, 1],
        [0.7071, 0.7071, 0.7071],
        [1.4142, 1.4142, 1.4142],
        [0.7071, 0.7071, 0.7071],
    ])
    expected_x2 = paddle.ones(5, 3)

    out = transform(data)
    assert len(out) == 4
    assert paddle.equal(out.edge_index, edge_index)
    assert paddle.allclose(out.x, x)
    assert paddle.allclose(out.x1, expected_x1, atol=1e-4)
    assert paddle.allclose(out.x2, expected_x2)

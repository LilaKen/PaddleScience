import pytest
import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import LinearTransformation


@pytest.mark.parametrize('matrix', [
    [[2.0, 0.0], [0.0, 2.0]],
    paddle.to_tensor([[2.0, 0.0], [0.0, 2.0]]),
])
def test_linear_transformation(matrix):
    pos = paddle.to_tensor([[-1.0, 1.0], [-3.0, 0.0], [2.0, -1.0]])

    transform = LinearTransformation(matrix)
    assert str(transform) == ('LinearTransformation(\n'
                              '[[2. 0.]\n'
                              ' [0. 2.]]\n'
                              ')')

    out = transform(Data(pos=pos))
    assert len(out) == 1
    assert paddle.allclose(out.pos, 2 * pos)

    out = transform(Data())
    assert len(out) == 0

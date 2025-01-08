import paddle

from paddle_geometric.testing import is_full_test
from paddle_geometric.utils import normalized_cut


def test_normalized_cut():
    row = paddle.to_tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    col = paddle.to_tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3])
    edge_attr = paddle.to_tensor(
        [3.0, 3.0, 6.0, 3.0, 6.0, 1.0, 3.0, 2.0, 1.0, 2.0])
    expected = paddle.to_tensor([4.0, 4.0, 5.0, 2.5, 5.0, 1.0, 2.5, 2.0, 1.0, 2.0])

    out = normalized_cut(paddle.stack([row, col], dim=0), edge_attr)
    assert paddle.allclose(out, expected)

    if is_full_test():
        jit = paddle.jit.to_static(normalized_cut)
        out = jit(paddle.stack([row, col], dim=0), edge_attr)
        assert paddle.allclose(out, expected)

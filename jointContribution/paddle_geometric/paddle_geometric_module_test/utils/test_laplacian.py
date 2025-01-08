import paddle

from paddle_geometric.testing import is_full_test
from paddle_geometric.utils import get_laplacian


def test_get_laplacian():
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=paddle.int64)
    edge_weight = paddle.to_tensor([1, 2, 2, 4], dtype=paddle.float32)

    lap = get_laplacian(edge_index, edge_weight)
    assert lap[0].tolist() == [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
    assert lap[1].tolist() == [-1.0, -2.0, -2.0, -4.0, 1.0, 2.0, 4.0]

    if is_full_test():
        jit = paddle.jit.to_static(get_laplacian)
        lap = jit(edge_index, edge_weight)
        assert lap[0].tolist() == [[0, 1, 1, 2, 0, 1, 2],
                                   [1, 0, 2, 1, 0, 1, 2]]
        assert lap[1].tolist() == [-1.0, -2.0, -2.0, -4.0, 1.0, 2.0, 4.0]

    lap_sym = get_laplacian(edge_index, edge_weight, normalization='sym')
    assert lap_sym[0].tolist() == lap[0].tolist()
    assert lap_sym[1].tolist() == [-0.7071067094802856, -1.4142134189605713, -0.7071067094802856, -1.4142134189605713, 1.0, 1.0, 1.0]

    lap_rw = get_laplacian(edge_index, edge_weight, normalization='rw')
    assert lap_rw[0].tolist() == lap[0].tolist()
    assert lap_rw[1].tolist() == [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]

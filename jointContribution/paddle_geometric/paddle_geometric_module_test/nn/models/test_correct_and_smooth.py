import paddle

import paddle_geometric.typing
from paddle_geometric.nn.models import CorrectAndSmooth
from paddle_geometric.testing import noWindows
from paddle_geometric.typing import SparseTensor


@noWindows
def test_correct_and_smooth():
    y_soft = paddle.to_tensor([0.1, 0.5, 0.4]).repeat(6, 1)
    y_true = paddle.to_tensor([1, 0, 0, 2, 1, 1])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = paddle.randint(0, 2, (6, ), dtype=paddle.bool)

    model = CorrectAndSmooth(
        num_correction_layers=2,
        correction_alpha=0.5,
        num_smoothing_layers=2,
        smoothing_alpha=0.5,
    )
    assert str(model) == ('CorrectAndSmooth(\n'
                          '  correct: num_layers=2, alpha=0.5\n'
                          '  smooth:  num_layers=2, alpha=0.5\n'
                          '  autoscale=True, scale=1.0\n'
                          ')')

    out = model.correct(y_soft, y_true[mask], mask, edge_index)
    assert out.shape== (6, 3)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert paddle.allclose(
            out, model.correct(y_soft, y_true[mask], mask, adj.t()))

    out = model.smooth(y_soft, y_true[mask], mask, edge_index)
    assert out.shape== (6, 3)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(
            out, model.smooth(y_soft, y_true[mask], mask, adj.t()))

    # Test without autoscale:
    model = CorrectAndSmooth(
        num_correction_layers=2,
        correction_alpha=0.5,
        num_smoothing_layers=2,
        smoothing_alpha=0.5,
        autoscale=False,
    )
    out = model.correct(y_soft, y_true[mask], mask, edge_index)
    assert out.shape== (6, 3)
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        assert paddle.allclose(
            out, model.correct(y_soft, y_true[mask], mask, adj.t()))

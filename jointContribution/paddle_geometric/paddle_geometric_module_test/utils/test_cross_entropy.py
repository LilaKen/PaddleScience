import pytest
import paddle
import paddle.nn.functional as F

from paddle_geometric.utils.cross_entropy import sparse_cross_entropy


@pytest.mark.parametrize('with_edge_label_weight', [False, True])
def test_sparse_cross_entropy_multiclass(with_edge_label_weight):
    x = paddle.randn(shape=[5, 5])
    x.stop_gradient = False
    y = paddle.eye(5)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = paddle.rand(edge_label_index.shape[1])
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y)
    expected.backward()
    expected_grad = x.grad

    x.grad = None
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert paddle.allclose(expected, out)
    assert paddle.allclose(expected_grad, x.grad)


@pytest.mark.parametrize('with_edge_label_weight', [False, True])
def test_sparse_cross_entropy_multilabel(with_edge_label_weight):
    x = paddle.randn(shape=[4, 4])
    x.stop_gradient = False
    y = paddle.randint_like(x, 0, 2)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = paddle.rand([edge_label_index.shape[1]])
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y, soft_label=True)
    expected.backward()
    expected_grad = x.grad

    x.grad = paddle.zeros_like(x.grad)  # Use zeros_like to clear gradients
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert paddle.allclose(expected, out)
    assert paddle.allclose(expected_grad, x.grad)

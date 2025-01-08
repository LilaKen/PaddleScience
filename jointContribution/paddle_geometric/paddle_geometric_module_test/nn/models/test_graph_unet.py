import paddle

from paddle_geometric.nn import GraphUNet
from paddle_geometric.testing import is_full_test, onlyLinux


@onlyLinux  # TODO  (matthias) Investigate CSR @ CSR support on Windows.
def test_graph_unet():
    model = GraphUNet(16, 32, 8, depth=3)
    out = 'GraphUNet(16, 32, 8, depth=3, pool_ratios=[0.5, 0.5, 0.5])'
    assert str(model) == out

    x = paddle.randn(shape=[3, 16])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = model(x, edge_index)
    assert out.shape== (3, 8)

    if is_full_test():
        jit = paddle.jit.export(model)
        out = jit(x, edge_index)
        assert out.shape== (3, 8)

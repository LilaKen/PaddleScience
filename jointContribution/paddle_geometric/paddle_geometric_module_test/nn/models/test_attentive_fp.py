import paddle

from paddle_geometric.nn import AttentiveFP
from paddle_geometric.testing import is_full_test


def test_attentive_fp():
    model = AttentiveFP(8, 16, 32, edge_dim=3, num_layers=2, num_timesteps=2)
    assert str(model) == ('AttentiveFP(in_channels=8, hidden_channels=16, '
                          'out_channels=32, edge_dim=3, num_layers=2, '
                          'num_timesteps=2)')

    x = paddle.randn(shape=[4, 8])
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 3])
    batch = paddle.to_tensor([0, 0, 0, 0])

    out = model(x, edge_index, edge_attr, batch)
    assert out.shape== (1, 32)

    if is_full_test():
        jit = paddle.jit.to_static(model)
        assert paddle.allclose(jit(x, edge_index, edge_attr, batch), out)

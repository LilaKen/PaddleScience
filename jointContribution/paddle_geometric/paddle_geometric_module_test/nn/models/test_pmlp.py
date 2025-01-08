import pytest
import paddle

from paddle_geometric.nn.models import PMLP


def test_pmlp():
    x = paddle.randn(shape=[4, 16])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    pmlp = PMLP(in_channels=16, hidden_channels=32, out_channels=2,
                num_layers=4)
    assert str(pmlp) == 'PMLP(16, 2, num_layers=4)'

    pmlp.training = True
    assert pmlp(x).shape== (4, 2)

    pmlp.training = False
    assert pmlp(x, edge_index).shape== (4, 2)

    with pytest.raises(ValueError, match="'edge_index' needs to be present"):
        pmlp.training = False
        pmlp(x)

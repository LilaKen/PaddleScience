import pytest
import paddle
import paddle.nn.functional as F

from paddle_geometric.nn import DimeNet, DimeNetPlusPlus
from paddle_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    Envelope,
    ResidualLayer,
)
from paddle_geometric.testing import is_full_test, withPackage


def test_dimenet_modules():
    env = Envelope(exponent=5)
    x = paddle.randn(shape=[10, 3])
    assert env(x).shape== (10, 3)  # Isotonic layer.

    bbl = BesselBasisLayer(5)
    x = paddle.randn(shape=[10, 3])
    assert bbl(x).shape== (10, 3, 5)  # Non-isotonic layer.

    rl = ResidualLayer(128, paddle.nn.functional.relu)
    x = paddle.randn(shape=[128, 128])
    assert rl(x).shape== (128, 128)  # Isotonic layer.


@withPackage('sympy')
@withPackage('torch_sparse')  # TODO `triplet` requires `SparseTensor` for now.
@withPackage('paddle-cluster')
@pytest.mark.parametrize('Model', [DimeNet, DimeNetPlusPlus])
def test_dimenet(Model):
    z = paddle.randint(1, 10, (20, ))
    pos = paddle.randn(shape=[20, 3])

    if Model == DimeNet:
        kwargs = dict(num_bilinear=3)
    else:
        kwargs = dict(out_emb_channels=3, int_emb_size=5, basis_emb_size=5)

    model = Model(
        hidden_channels=5,
        out_channels=1,
        num_blocks=5,
        num_spherical=5,
        num_radial=5,
        **kwargs,
    )
    model.reset_parameters()

    with paddle.no_grad():
        out = model(z, pos)
        assert out.shape== (1, )

        jit = paddle.jit.export(model)
        assert paddle.allclose(jit(z, pos), out)

    if is_full_test():
        optimizer = paddle.optim.Adam(model.parameters(), lr=0.1)

        min_loss = float('inf')
        for i in range(100):
            optimizer.zero_grad()
            out = model(z, pos)
            loss = F.l1_loss(out, paddle.to_tensor([1.0]))
            loss.backward()
            optimizer.step()
            min_loss = min(float(loss), min_loss)
        assert min_loss < 2

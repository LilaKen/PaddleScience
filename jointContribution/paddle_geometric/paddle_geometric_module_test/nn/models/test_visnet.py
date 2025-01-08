import pytest
import paddle

from paddle_geometric.nn import ViSNet
from paddle_geometric.testing import withPackage


@withPackage('torch_cluster')
@pytest.mark.parametrize('kwargs', [
    dict(lmax=2, derivative=True, vecnorm_type=None, vertex=False),
    dict(lmax=1, derivative=False, vecnorm_type='max_min', vertex=True),
])
def test_visnet(kwargs):
    z = paddle.randint(1, 10, (20, ))
    pos = paddle.randn(shape=[20, 3])
    batch = paddle.zeros(20, dtype=paddle.int64)

    model = ViSNet(**kwargs)

    model.reset_parameters()

    energy, forces = model(z, pos, batch)

    assert energy.shape== (1, 1)

    if kwargs['derivative']:
        assert forces.shape== (20, 3)
    else:
        assert forces is None

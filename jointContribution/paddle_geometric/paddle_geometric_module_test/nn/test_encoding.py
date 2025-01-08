import paddle

from paddle_geometric.nn import PositionalEncoding, TemporalEncoding
from paddle_geometric.testing import withDevice


@withDevice
def test_positional_encoding(device):
    encoder = PositionalEncoding(64).to(device)
    assert str(encoder) == 'PositionalEncoding(64)'

    x = paddle.to_tensor([1.0, 2.0, 3.0], place=device)
    assert encoder(x).shape == [3, 64]


@withDevice
def test_temporal_encoding(device):
    encoder = TemporalEncoding(64).to(device)
    assert str(encoder) == 'TemporalEncoding(64)'

    x = paddle.to_tensor([1.0, 2.0, 3.0], place=device)
    assert encoder(x).shape == [3, 64]

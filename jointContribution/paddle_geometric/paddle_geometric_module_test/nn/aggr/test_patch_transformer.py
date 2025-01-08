import paddle

from paddle_geometric.nn import PatchTransformerAggregation
from paddle_geometric.testing import withCUDA


@withCUDA
def test_patch_transformer_aggregation(device: paddle.place) -> None:
    aggr = PatchTransformerAggregation(
        in_channels=16,
        out_channels=32,
        patch_size=2,
        hidden_channels=8,
        num_transformer_blocks=1,
        heads=2,
        dropout=0.2,
        aggr=['sum', 'mean', 'min', 'max', 'var', 'std'],
    ).to(device)
    aggr.reset_parameters()
    assert str(aggr) == 'PatchTransformerAggregation(16, 32, patch_size=2)'

    index = paddle.to_tensor([0, 0, 1, 1, 1, 2], place=device)
    x = paddle.randn(shape=[index.shape[0], 16, place=device])

    out = aggr(x, index)
    assert out.device == device
    assert out.shape== (3, aggr.out_channels)

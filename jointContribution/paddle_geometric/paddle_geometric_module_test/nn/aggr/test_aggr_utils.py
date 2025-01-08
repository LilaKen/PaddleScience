import paddle

from paddle_geometric.nn.aggr.utils import (
    InducedSetAttentionBlock,
    MultiheadAttentionBlock,
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)


def test_multihead_attention_block():
    x = paddle.randn(shape=[2, 4, 8])
    y = paddle.randn(shape=[2, 3, 8])
    x_mask = paddle.to_tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=paddle.bool)
    y_mask = paddle.to_tensor([[1, 1, 0], [1, 1, 1]], dtype=paddle.bool)

    block = MultiheadAttentionBlock(8, heads=2)
    block.reset_parameters()
    assert str(block) == ('MultiheadAttentionBlock(8, heads=2, '
                          'layer_norm=True, dropout=0.0)')

    out = block(x, y, x_mask, y_mask)
    assert out.shape == [2, 4, 8]

    jit = paddle.jit.to_static(block)
    assert paddle.allclose(jit(x, y, x_mask, y_mask), out)


def test_multihead_attention_block_dropout():
    x = paddle.randn(shape=[2, 4, 8])

    block = MultiheadAttentionBlock(8, dropout=0.5)
    assert not paddle.allclose(block(x, x), block(x, x))


def test_set_attention_block():
    x = paddle.randn(shape=[2, 4, 8])
    mask = paddle.to_tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=paddle.bool)

    block = SetAttentionBlock(8, heads=2)
    block.reset_parameters()
    assert str(block) == ('SetAttentionBlock(8, heads=2, layer_norm=True, '
                          'dropout=0.0)')

    out = block(x, mask)
    assert out.shape == [2, 4, 8]

    jit = paddle.jit.to_static(block)
    assert paddle.allclose(jit(x, mask), out)


def test_induced_set_attention_block():
    x = paddle.randn(shape=[2, 4, 8])
    mask = paddle.to_tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=paddle.bool)

    block = InducedSetAttentionBlock(8, num_induced_points=2, heads=2)
    assert str(block) == ('InducedSetAttentionBlock(8, num_induced_points=2, '
                          'heads=2, layer_norm=True, dropout=0.0)')

    out = block(x, mask)
    assert out.shape == [2, 4, 8]

    jit = paddle.jit.to_static(block)
    assert paddle.allclose(jit(x, mask), out)


def test_pooling_by_multihead_attention():
    x = paddle.randn(shape=[2, 4, 8])
    mask = paddle.to_tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=paddle.bool)

    block = PoolingByMultiheadAttention(8, num_seed_points=2, heads=2)
    assert str(block) == ('PoolingByMultiheadAttention(8, num_seed_points=2, '
                          'heads=2, layer_norm=True, dropout=0.0)')

    out = block(x, mask)
    assert out.shape == [2, 2, 8]

    jit = paddle.jit.to_static(block)
    assert paddle.allclose(jit(x, mask), out)

import pytest
import paddle
from paddle.nn import ReLU

from paddle_geometric.nn import DeepGCNLayer, GENConv, LayerNorm


@pytest.mark.parametrize(
    'block_tuple',
    [('res+', 1), ('res', 1), ('dense', 2), ('plain', 1)],
)
@pytest.mark.parametrize('ckpt_grad', [True, False])
def test_deepgcn(block_tuple, ckpt_grad):
    block, expansion = block_tuple
    x = paddle.randn(shape=[3, 8])
    edge_index = paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    conv = GENConv(8, 8)
    norm = LayerNorm(8)
    act = ReLU()
    layer = DeepGCNLayer(conv, norm, act, block=block, ckpt_grad=ckpt_grad)
    assert str(layer) == f'DeepGCNLayer(block={block})'

    out = layer(x, edge_index)
    assert out.shape== (3, 8 * expansion)

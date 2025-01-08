import pytest
import paddle

from paddle_geometric.nn.conv import SAGEConv
from paddle_geometric.nn.dense import Linear
from paddle_geometric.nn.to_hetero_module import (
    ToHeteroLinear,
    ToHeteroMessagePassing,
)


@pytest.mark.parametrize('LinearCls', [paddle.nn.Linear, Linear])
def test_to_hetero_linear(LinearCls):
    x_dict = {'1': paddle.randn(shape=[5, 16]), '2': paddle.randn(shape=[4, 16])}
    x = paddle.concat([x_dict['1'], x_dict['2']], dim=0)
    type_vec = paddle.to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

    module = ToHeteroLinear(LinearCls(16, 32), list(x_dict.keys()))

    out_dict = module(x_dict)
    assert len(out_dict) == 2
    assert out_dict['1'].shape== (5, 32)
    assert out_dict['2'].shape== (4, 32)

    out = module(x, type_vec)
    assert out.shape== (9, 32)

    assert paddle.allclose(out_dict['1'], out[0:5])
    assert paddle.allclose(out_dict['2'], out[5:9])


def test_to_hetero_message_passing():
    x_dict = {'1': paddle.randn(shape=[5, 16]), '2': paddle.randn(shape=[4, 16])}
    x = paddle.concat([x_dict['1'], x_dict['2']], dim=0)
    node_type = paddle.to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

    edge_index_dict = {
        ('1', 'to', '2'): paddle.to_tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]]),
        ('2', 'to', '1'): paddle.to_tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]]),
    }
    edge_index = paddle.to_tensor([
        [0, 1, 2, 3, 4, 5, 5, 6, 7, 8],
        [5, 5, 6, 7, 8, 0, 1, 2, 3, 4],
    ])
    edge_type = paddle.to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    module = ToHeteroMessagePassing(SAGEConv(16, 32), list(x_dict.keys()),
                                    list(edge_index_dict.keys()))

    out_dict = module(x_dict, edge_index_dict)
    assert len(out_dict) == 2
    assert out_dict['1'].shape== (5, 32)
    assert out_dict['2'].shape== (4, 32)

    out = module(x, edge_index, node_type, edge_type)
    assert out.shape== (9, 32)

    assert paddle.allclose(out_dict['1'], out[0:5])
    assert paddle.allclose(out_dict['2'], out[5:9])

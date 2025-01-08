import paddle

import paddle_geometric.graphgym.register as register
from paddle_geometric.testing import withPackage


@register.register_act('identity')
def identity_act(x: paddle.to_tensor) -> paddle.to_tensor:
    return x


@withPackage('yacs')
def test_register():
    assert len(register.act_dict) == 8
    assert list(register.act_dict.keys()) == [
        'relu', 'selu', 'prelu', 'elu', 'lrelu_01', 'lrelu_025', 'lrelu_05',
        'identity'
    ]
    assert str(register.act_dict['relu']()) == 'ReLU()'

    register.register_act('lrelu_03', paddle.nn.LeakyReLU(0.3))
    assert len(register.act_dict) == 9
    assert 'lrelu_03' in register.act_dict

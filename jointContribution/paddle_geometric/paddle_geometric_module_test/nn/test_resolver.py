import pytest
import paddle
from paddle.optim.lr_scheduler import ConstantLR, LambdaLR, ReduceLROnPlateau

import paddle_geometric
from paddle_geometric.nn.resolver import (
    activation_resolver,
    aggregation_resolver,
    lr_scheduler_resolver,
    normalization_resolver,
    optimizer_resolver,
)


def test_activation_resolver():
    assert isinstance(activation_resolver(paddle.nn.ELU()), paddle.nn.ELU)
    assert isinstance(activation_resolver(paddle.nn.ReLU()), paddle.nn.ReLU)
    assert isinstance(activation_resolver(paddle.nn.PReLU()), paddle.nn.PReLU)

    assert isinstance(activation_resolver('elu'), paddle.nn.ELU)
    assert isinstance(activation_resolver('relu'), paddle.nn.ReLU)
    assert isinstance(activation_resolver('prelu'), paddle.nn.PReLU)


@pytest.mark.parametrize('aggr_tuple', [
    (paddle_geometric.nn.MeanAggregation, 'mean'),
    (paddle_geometric.nn.SumAggregation, 'sum'),
    (paddle_geometric.nn.SumAggregation, 'add'),
    (paddle_geometric.nn.MaxAggregation, 'max'),
    (paddle_geometric.nn.MinAggregation, 'min'),
    (paddle_geometric.nn.MulAggregation, 'mul'),
    (paddle_geometric.nn.VarAggregation, 'var'),
    (paddle_geometric.nn.StdAggregation, 'std'),
    (paddle_geometric.nn.SoftmaxAggregation, 'softmax'),
    (paddle_geometric.nn.PowerMeanAggregation, 'powermean'),
])
def test_aggregation_resolver(aggr_tuple):
    aggr_module, aggr_repr = aggr_tuple
    assert isinstance(aggregation_resolver(aggr_module()), aggr_module)
    assert isinstance(aggregation_resolver(aggr_repr), aggr_module)


def test_multi_aggregation_resolver():
    aggr = aggregation_resolver(None)
    assert aggr is None

    aggr = aggregation_resolver(['sum', 'mean', None])
    assert isinstance(aggr, paddle_geometric.nn.MultiAggregation)
    assert len(aggr.aggrs) == 3
    assert isinstance(aggr.aggrs[0], paddle_geometric.nn.SumAggregation)
    assert isinstance(aggr.aggrs[1], paddle_geometric.nn.MeanAggregation)
    assert aggr.aggrs[2] is None


@pytest.mark.parametrize('norm_tuple', [
    (paddle_geometric.nn.BatchNorm, 'batch', (16, )),
    (paddle_geometric.nn.BatchNorm, 'batch_norm', (16, )),
    (paddle_geometric.nn.InstanceNorm, 'instance_norm', (16, )),
    (paddle_geometric.nn.LayerNorm, 'layer_norm', (16, )),
    (paddle_geometric.nn.GraphNorm, 'graph_norm', (16, )),
    (paddle_geometric.nn.GraphSizeNorm, 'graphsize_norm', ()),
    (paddle_geometric.nn.PairNorm, 'pair_norm', ()),
    (paddle_geometric.nn.MessageNorm, 'message_norm', ()),
    (paddle_geometric.nn.DiffGroupNorm, 'diffgroup_norm', (16, 4)),
])
def test_normalization_resolver(norm_tuple):
    norm_module, norm_repr, norm_args = norm_tuple
    assert isinstance(normalization_resolver(norm_module(*norm_args)),
                      norm_module)
    assert isinstance(normalization_resolver(norm_repr, *norm_args),
                      norm_module)


def test_optimizer_resolver():
    params = [paddle.nn.Parameter(paddle.randn(shape=[1]))]

    assert isinstance(optimizer_resolver(paddle.optim.SGD(params, lr=0.01)),
                      paddle.optim.SGD)
    assert isinstance(optimizer_resolver(paddle.optim.Adam(params)),
                      paddle.optim.Adam)
    assert isinstance(optimizer_resolver(paddle.optim.Rprop(params)),
                      paddle.optim.Rprop)

    assert isinstance(optimizer_resolver('sgd', params, lr=0.01),
                      paddle.optim.SGD)
    assert isinstance(optimizer_resolver('adam', params), paddle.optim.Adam)
    assert isinstance(optimizer_resolver('rprop', params), paddle.optim.Rprop)


@pytest.mark.parametrize('scheduler_args', [
    ('constant_with_warmup', LambdaLR),
    ('linear_with_warmup', LambdaLR),
    ('cosine_with_warmup', LambdaLR),
    ('cosine_with_warmup_restarts', LambdaLR),
    ('polynomial_with_warmup', LambdaLR),
    ('constant', ConstantLR),
    ('ReduceLROnPlateau', ReduceLROnPlateau),
])
def test_lr_scheduler_resolver(scheduler_args):
    scheduler_name, scheduler_cls = scheduler_args

    model = paddle.nn.Linear(10, 5)
    optimizer = paddle.optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = lr_scheduler_resolver(
        scheduler_name,
        optimizer,
        num_training_steps=100,
    )
    assert isinstance(lr_scheduler, scheduler_cls)

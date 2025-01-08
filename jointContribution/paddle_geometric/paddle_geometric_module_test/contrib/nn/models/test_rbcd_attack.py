import pytest
import paddle
from paddle.nn import Linear

from paddle_geometric.contrib.nn import GRBCDAttack, PRBCDAttack
from paddle_geometric.nn import GCNConv, global_add_pool
from paddle_geometric.utils import to_undirected
from paddle.nn.functional import relu, sigmoid, log_softmax


class GCN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 3)
        self.conv2 = GCNConv(7, 16)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GNN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 3)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(7, 16)

    def forward(self, x, edge_index, edge_weight, batch=None):
        x = relu(self.conv1(x, edge_index, edge_weight))
        x = relu(self.conv2(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x


@pytest.mark.parametrize('model', [GCN, GNN])
@pytest.mark.parametrize('budget', [1])
@pytest.mark.parametrize('loss',
                         ['masked', 'margin', 'prob_margin', 'tanh_margin'])
@pytest.mark.parametrize('is_undirected', [False, True])
@pytest.mark.parametrize('with_early_stopping', [False, True])
def test_prbcd_attack(model, budget, loss, is_undirected, with_early_stopping):
    attack = PRBCDAttack(model(), block_size=10_000, epochs=4,
                         epochs_resampling=2, loss=loss, max_final_samples=2,
                         log=False, is_undirected=is_undirected,
                         with_early_stopping=with_early_stopping)
    assert str(attack) == 'PRBCDAttack()'

    x = paddle.randn(shape=[8, 3])
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
    if is_undirected:
        edge_index = to_undirected(edge_index)

    if model == GNN:
        y = paddle.to_tensor([0])
        # All nodes belong to same graph:
        kwargs = dict(batch=edge_index.new_zeros(x.shape[0]))
    else:
        y = paddle.to_tensor([0, 1, 1, 0, 1, 0, 1, 0])
        kwargs = {}

    pert_edge_index, pert = attack.attack(x, edge_index, y, budget, **kwargs)

    m = edge_index.shape[1]
    if budget == 1:
        assert pert.shapein [(2, 0), (2, 1)]

        if pert.shape[1]:
            if is_undirected:
                possible_m = [m - 2, m + 2]
            else:
                possible_m = [m - 1, m + 1]
        else:
            possible_m = [m]
        assert pert_edge_index.shape[1] in possible_m


@pytest.mark.parametrize('model', [GCN, GNN])
@pytest.mark.parametrize('budget', [1])
@pytest.mark.parametrize('is_undirected', [False, True])
def test_grbcd_attack(model, budget, is_undirected):
    attack = GRBCDAttack(model(), block_size=10_000, epochs=4, log=False,
                         is_undirected=is_undirected)
    assert str(attack) == 'GRBCDAttack()'

    x = paddle.randn(shape=[8, 3], dtype=paddle.float32)
    edge_index = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=paddle.float32)
    if is_undirected:
        edge_index = to_undirected(edge_index)

    if model == GNN:
        y = paddle.to_tensor([0], dtype=paddle.float32)
        # All nodes belong to same graph:
        kwargs = dict(batch=edge_index.new_zeros(x.shape[0]))
    else:
        y = paddle.to_tensor([0, 1, 1, 0, 1, 0, 1, 0], dtype=paddle.float32)
        kwargs = {}

    pert_edge_index, pert = attack.attack(x, edge_index, y, budget, **kwargs)

    m = edge_index.shape[1]
    if budget == 1:
        assert pert.shape == [2, 1]

        if is_undirected:
            possible_m = [m - 2, m + 2]
        else:
            possible_m = [m - 1, m + 1]
        assert pert_edge_index.shape[1] in possible_m

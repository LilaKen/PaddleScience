from typing import Optional

import pytest
import paddle
from paddle import Tensor, nn

import paddle_geometric.typing
from paddle_geometric.nn import Linear, SAGEConv, summary, to_hetero
from paddle_geometric.nn.models import GCN
from paddle_geometric.testing import withPackage
from paddle_geometric.typing import SparseTensor


class GraphSAGE(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 16)
        self.conv1 = SAGEConv(16, 32)
        self.lin2 = Linear(32, 32)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.lin1(x).relu()
        x = self.conv1(x, edge_index).relu()
        x = self.lin2(x)
        return x


class ModuleDictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.acts = nn.ModuleDict({
            "lrelu": nn.LeakyReLU(),
            "prelu": nn.PReLU()
        })

    def forward(self, x: paddle.to_tensor, act_type: str) -> paddle.to_tensor:
        return self.acts[act_type](x)


@pytest.fixture
def gcn():
    paddle.seed(1)
    model = GCN(32, 16, num_layers=2, out_channels=32)
    x = paddle.randn(shape=[100, 32])
    edge_index = paddle.randint(100, size=(2, 20))
    adj_t: Optional[SparseTensor] = None
    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj_t = SparseTensor.from_edge_index(
            edge_index,
            sparse_sizes=(100, 100),
        ).t()
    return dict(model=model, x=x, edge_index=edge_index, adj_t=adj_t)


@withPackage('tabulate')
def test_summary_basic(gcn):
    expected = """
+---------------------+--------------------+----------------+----------+
| Layer               | Input Shape        | Output Shape   | #Param   |
|---------------------+--------------------+----------------+----------|
| GCN                 | [100, 32], [2, 20] | [100, 32]      | 1,072    |
| ├─(dropout)Dropout  | [100, 16]          | [100, 16]      | --       |
| ├─(act)ReLU         | [100, 16]          | [100, 16]      | --       |
| ├─(convs)ModuleList | --                 | --             | 1,072    |
| │    └─(0)GCNConv   | [100, 32], [2, 20] | [100, 16]      | 528      |
| │    └─(1)GCNConv   | [100, 16], [2, 20] | [100, 32]      | 544      |
| ├─(norms)ModuleList | --                 | --             | --       |
| │    └─(0)Identity  | [100, 16]          | [100, 16]      | --       |
| │    └─(1)Identity  | --                 | --             | --       |
+---------------------+--------------------+----------------+----------+
"""
    assert summary(gcn['model'], gcn['x'], gcn['edge_index']) == expected[1:-1]


@withPackage('tabulate', 'paddle_sparse')
def test_summary_with_sparse_tensor(gcn):
    expected = """
+---------------------+-----------------------+----------------+----------+
| Layer               | Input Shape           | Output Shape   | #Param   |
|---------------------+-----------------------+----------------+----------|
| GCN                 | [100, 32], [100, 100] | [100, 32]      | 1,072    |
| ├─(dropout)Dropout  | [100, 16]             | [100, 16]      | --       |
| ├─(act)ReLU         | [100, 16]             | [100, 16]      | --       |
| ├─(convs)ModuleList | --                    | --             | 1,072    |
| │    └─(0)GCNConv   | [100, 32], [100, 100] | [100, 16]      | 528      |
| │    └─(1)GCNConv   | [100, 16], [100, 100] | [100, 32]      | 544      |
| ├─(norms)ModuleList | --                    | --             | --       |
| │    └─(0)Identity  | [100, 16]             | [100, 16]      | --       |
| │    └─(1)Identity  | --                    | --             | --       |
+---------------------+-----------------------+----------------+----------+
"""
    assert summary(gcn['model'], gcn['x'], gcn['adj_t']) == expected[1:-1]


@withPackage('tabulate')
def test_lazy_gcn():
    expected = """
+---------------------+--------------------+----------------+----------+
| Layer               | Input Shape        | Output Shape   | #Param   |
|---------------------+--------------------+----------------+----------|
| GCN                 | [100, 32], [2, 20] | [100, 32]      | -1       |
| ├─(dropout)Dropout  | [100, 16]          | [100, 16]      | --       |
| ├─(act)ReLU         | [100, 16]          | [100, 16]      | --       |
| ├─(convs)ModuleList | --                 | --             | -1       |
| │    └─(0)GCNConv   | [100, 32], [2, 20] | [100, 16]      | -1       |
| │    └─(1)GCNConv   | [100, 16], [2, 20] | [100, 32]      | 544      |
| ├─(norms)ModuleList | --                 | --             | --       |
| │    └─(0)Identity  | [100, 16]          | [100, 16]      | --       |
| │    └─(1)Identity  | --                 | --             | --       |
+---------------------+--------------------+----------------+----------+
"""
    model = GCN(-1, 16, num_layers=2, out_channels=32)
    x = paddle.randn(shape=[100, 32])
    edge_index = paddle.randint(100, size=(2, 20))

    assert summary(model, x, edge_index) == expected[1:-1]


@withPackage('tabulate')
def test_summary_with_max_depth(gcn):
    expected = """
+---------------------+--------------------+----------------+----------+
| Layer               | Input Shape        | Output Shape   | #Param   |
|---------------------+--------------------+----------------+----------|
| GCN                 | [100, 32], [2, 20] | [100, 32]      | 1,072    |
| ├─(dropout)Dropout  | [100, 16]          | [100, 16]      | --       |
| ├─(act)ReLU         | [100, 16]          | [100, 16]      | --       |
| ├─(convs)ModuleList | --                 | --             | 1,072    |
| ├─(norms)ModuleList | --                 | --             | --       |
+---------------------+--------------------+----------------+----------+
"""
    assert summary(
        gcn['model'],
        gcn['x'],
        gcn['edge_index'],
        max_depth=1,
    ) == expected[1:-1]


@withPackage('tabulate')
def test_summary_with_leaf_module(gcn):
    expected = """# noqa: E501
+-----------------------------------------+--------------------+----------------+----------+
| Layer                                   | Input Shape        | Output Shape   | #Param   |
|-----------------------------------------+--------------------+----------------+----------|
| GCN                                     | [100, 32], [2, 20] | [100, 32]      | 1,072    |
| ├─(dropout)Dropout                      | [100, 16]          | [100, 16]      | --       |
| ├─(act)ReLU                             | [100, 16]          | [100, 16]      | --       |
| ├─(convs)ModuleList                     | --                 | --             | 1,072    |
| │    └─(0)GCNConv                       | [100, 32], [2, 20] | [100, 16]      | 528      |
| │    │    └─(aggr_module)SumAggregation | [120, 16]          | [100, 16]      | --       |
| │    │    └─(lin)Linear                 | [100, 32]          | [100, 16]      | 512      |
| │    └─(1)GCNConv                       | [100, 16], [2, 20] | [100, 32]      | 544      |
| │    │    └─(aggr_module)SumAggregation | [120, 32]          | [100, 32]      | --       |
| │    │    └─(lin)Linear                 | [100, 16]          | [100, 32]      | 512      |
| ├─(norms)ModuleList                     | --                 | --             | --       |
| │    └─(0)Identity                      | [100, 16]          | [100, 16]      | --       |
| │    └─(1)Identity                      | --                 | --             | --       |
+-----------------------------------------+--------------------+----------------+----------+
"""
    assert summary(
        gcn['model'],
        gcn['x'],
        gcn['edge_index'],
        leaf_module=None,
    ) == expected[13:-1]


@withPackage('tabulate')
def test_summary_with_reusing_layers():
    act = nn.ReLU(inplace=True)
    model1 = nn.Sequential(act, nn.Identity(), act, nn.Identity(), act)
    model2 = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Identity(),
        nn.ReLU(inplace=True),
        nn.Identity(),
        nn.ReLU(inplace=True),
    )
    x = paddle.randn(shape=[10])

    assert summary(model1, x) == summary(model2, x)


@withPackage('tabulate')
def test_summary_with_to_hetero_model():
    x_dict = {
        'p': paddle.randn(shape=[100, 16]),
        'a': paddle.randn(shape=[100, 16]),
    }
    edge_index_dict = {
        ('p', 'to', 'p'): paddle.randint(100, (2, 200)),
        ('p', 'to', 'a'): paddle.randint(100, (2, 200)),
        ('a', 'to', 'p'): paddle.randint(100, (2, 200)),
    }
    metadata = list(x_dict.keys()), list(edge_index_dict.keys())
    model = to_hetero(GraphSAGE(), metadata)

    expected = """
+---------------------------+---------------------+----------------+----------+
| Layer                     | Input Shape         | Output Shape   | #Param   |
|---------------------------+---------------------+----------------+----------|
| GraphModule               |                     |                | 5,824    |
| ├─(lin1)ModuleDict        | --                  | --             | 544      |
| │    └─(p)Linear          | [100, 16]           | [100, 16]      | 272      |
| │    └─(a)Linear          | [100, 16]           | [100, 16]      | 272      |
| ├─(conv1)ModuleDict       | --                  | --             | 3,168    |
| │    └─(p__to__p)SAGEConv | [100, 16], [2, 200] | [100, 32]      | 1,056    |
| │    └─(p__to__a)SAGEConv | [2, 200]            | [100, 32]      | 1,056    |
| │    └─(a__to__p)SAGEConv | [2, 200]            | [100, 32]      | 1,056    |
| ├─(lin2)ModuleDict        | --                  | --             | 2,112    |
| │    └─(p)Linear          | [100, 32]           | [100, 32]      | 1,056    |
| │    └─(a)Linear          | [100, 32]           | [100, 32]      | 1,056    |
+---------------------------+---------------------+----------------+----------+
"""
    assert summary(model, x_dict, edge_index_dict) == expected[1:-1]


@withPackage('tabulate')
def test_summary_with_module_dict_model():
    model = ModuleDictModel()
    x = paddle.randn(shape=[100, 32])

    expected = """
+-------------------------+---------------+----------------+----------+
| Layer                   | Input Shape   | Output Shape   | #Param   |
|-------------------------+---------------+----------------+----------|
| ModuleDictModel         | [100, 32]     | [100, 32]      | 1        |
| ├─(acts)ModuleDict      | --            | --             | 1        |
| │    └─(lrelu)LeakyReLU | --            | --             | --       |
| │    └─(prelu)PReLU     | [100, 32]     | [100, 32]      | 1        |
+-------------------------+---------------+----------------+----------+
"""
    assert summary(model, x, 'prelu') == expected[1:-1]


@withPackage('tabulate')
def test_summary_with_jit_model():
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
    model = paddle.jit.to_static(model)
    x = paddle.randn(shape=[100, 32])

    expected = """
+----------------------------+---------------+----------------+----------+
| Layer                      | Input Shape   | Output Shape   | #Param   |
|----------------------------+---------------+----------------+----------|
| RecursiveScriptModule      | --            | --             | 664      |
| ├─(0)RecursiveScriptModule | --            | --             | 528      |
| ├─(1)RecursiveScriptModule | --            | --             | --       |
| ├─(2)RecursiveScriptModule | --            | --             | 136      |
+----------------------------+---------------+----------------+----------+
"""
    assert summary(model, x) == expected[1:-1]

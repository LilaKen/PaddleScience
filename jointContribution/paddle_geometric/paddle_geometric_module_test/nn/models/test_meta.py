from typing import Optional

import paddle
from paddle import Tensor
from paddle.nn import Linear as Lin
from paddle.nn import ReLU
from paddle.nn import Sequential as Seq

from paddle_geometric.nn import MetaLayer
from paddle_geometric.testing import is_full_test
from paddle_geometric.utils import scatter

count = 0


def test_meta_layer():
    assert str(MetaLayer()) == ('MetaLayer(\n'
                                '  edge_model=None,\n'
                                '  node_model=None,\n'
                                '  global_model=None\n'
                                ')')

    def dummy_model(*args):
        global count
        count += 1
        return None

    x = paddle.randn(shape=[20, 10])
    edge_index = paddle.randint(0, high=10, size=(2, 20), dtype=paddle.int64)

    for edge_model in (dummy_model, None):
        for node_model in (dummy_model, None):
            for global_model in (dummy_model, None):
                model = MetaLayer(edge_model, node_model, global_model)
                out = model(x, edge_index)
                assert isinstance(out, tuple) and len(out) == 3

    assert count == 12


def test_meta_layer_example():
    class EdgeModel(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.edge_mlp = Seq(Lin(2 * 10 + 5 + 20, 5), ReLU(), Lin(5, 5))

        def forward(
            self,
            src: Tensor,
            dst: Tensor,
            edge_attr: Optional[Tensor],
            u: Optional[Tensor],
            batch: Optional[Tensor],
        ) -> Tensor:
            assert edge_attr is not None
            assert u is not None
            assert batch is not None
            out = paddle.concat([src, dst, edge_attr, u[batch]], 1)
            return self.edge_mlp(out)

    class NodeModel(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.node_mlp_1 = Seq(Lin(15, 10), ReLU(), Lin(10, 10))
            self.node_mlp_2 = Seq(Lin(2 * 10 + 20, 10), ReLU(), Lin(10, 10))

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor],
            u: Optional[Tensor],
            batch: Optional[Tensor],
        ) -> Tensor:
            assert edge_attr is not None
            assert u is not None
            assert batch is not None
            row = edge_index[0]
            col = edge_index[1]
            out = paddle.concat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter(out, col, dim=0, dim_size=x.shape[0], reduce='mean')
            out = paddle.concat([x, out, u[batch]], dim=1)
            return self.node_mlp_2(out)

    class GlobalModel(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.global_mlp = Seq(Lin(20 + 10, 20), ReLU(), Lin(20, 20))

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor],
            u: Optional[Tensor],
            batch: Optional[Tensor],
        ) -> Tensor:
            assert u is not None
            assert batch is not None
            out = paddle.concat([
                u,
                scatter(x, batch, dim=0, reduce='mean'),
            ], dim=1)
            return self.global_mlp(out)

    op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())

    x = paddle.randn(shape=[20, 10])
    edge_attr = paddle.randn(shape=[40, 5])
    u = paddle.randn(shape=[2, 20])
    batch = paddle.to_tensor([0] * 10 + [1] * 10)
    edge_index = paddle.randint(0, high=10, size=(2, 20), dtype=paddle.int64)
    edge_index = paddle.concat([edge_index, 10 + edge_index], dim=1)

    x_out, edge_attr_out, u_out = op(x, edge_index, edge_attr, u, batch)
    assert x_out.shape== (20, 10)
    assert edge_attr_out.shape== (40, 5)
    assert u_out.shape== (2, 20)

    if is_full_test():
        jit = paddle.jit.to_static(op)

        x_out, edge_attr_out, u_out = jit(x, edge_index, edge_attr, u, batch)
        assert x_out.shape== (20, 10)
        assert edge_attr_out.shape== (40, 5)
        assert u_out.shape== (2, 20)

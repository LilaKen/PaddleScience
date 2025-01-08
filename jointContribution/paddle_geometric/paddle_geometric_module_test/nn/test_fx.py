import paddle
import paddle.nn.functional as F
from paddle import Tensor


def test_dropout():
    class MyModule(paddle.nn.Layer):
        def forward(self, x: Tensor) -> Tensor:
            return F.dropout(x, p=1.0, training=self.training)

    module = MyModule()
    # torch.fx.symbolic_trace		功能缺失	有对应相近功能但设计差异大无法映射，一般无需新增
    graph_module = paddle.fx.symbolic_trace(module)
    graph_module.recompile()

    x = paddle.randn(shape=[4])

    graph_module.train()
    assert paddle.allclose(graph_module(x), paddle.zeros_like(x))

    # This is certainly undesired behavior due to tracing :(
    graph_module.eval()
    assert paddle.allclose(graph_module(x), paddle.zeros_like(x))

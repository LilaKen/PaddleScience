import paddle

from paddle_geometric import is_paddle_instance
from paddle_geometric.testing import onlyLinux


def test_basic():
    assert is_paddle_instance(paddle.nn.Linear(1, 1), paddle.nn.Linear)


@onlyLinux
def test_compile():
    model = paddle.jit.to_static(paddle.nn.Linear(1, 1))
    assert not isinstance(model, paddle.nn.Linear)
    assert is_paddle_instance(model, paddle.nn.Linear)

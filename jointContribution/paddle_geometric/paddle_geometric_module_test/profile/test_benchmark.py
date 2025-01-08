import paddle

from paddle_geometric.profile import benchmark
from paddle_geometric.testing import withPackage


@withPackage('tabulate')
def test_benchmark(capfd):
    def add(x, y):
        return x + y

    benchmark(
        funcs=[add],
        args=(paddle.randn(shape=[10]), paddle.randn(shape=[10])),
        num_steps=1,
        num_warmups=1,
        backward=True,
    )

    out, _ = capfd.readouterr()
    assert '| Name   | Forward   | Backward   | Total   |' in out
    assert '| add    |' in out

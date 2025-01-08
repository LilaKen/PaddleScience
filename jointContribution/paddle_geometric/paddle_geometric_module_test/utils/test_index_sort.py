import paddle

from paddle_geometric.testing import withDevice
from paddle_geometric.utils import index_sort


@withDevice
def test_index_sort_stable(device):
    for _ in range(100):
        inputs = paddle.randint(0, 4, shape=[10, ])

        out = index_sort(inputs, stable=True)
        expected = paddle.sort(inputs, stable=True)

        assert paddle.equal_all(out, expected)

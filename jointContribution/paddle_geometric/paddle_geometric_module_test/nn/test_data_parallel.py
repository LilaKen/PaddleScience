import pytest
import paddle

from paddle_geometric.data import Data
from paddle_geometric.nn import DataParallel
from paddle_geometric.testing import onlyCUDA


@onlyCUDA
def test_data_parallel_single_gpu():
    with pytest.warns(UserWarning, match="much slower"):
        module = DataParallel(paddle.nn.Identity())
    data_list = [Data(x=paddle.randn(shape=[x, 1])) for x in [2, 3, 10, 4]]
    batches = module.scatter(data_list, device_ids=[0])
    assert len(batches) == 1


@onlyCUDA
@pytest.mark.skipif(paddle.device.cuda.device_count() < 2, reason='No multiple GPUs')
def test_data_parallel_multi_gpu():
    with pytest.warns(UserWarning, match="much slower"):
        module = DataParallel(paddle.nn.Identity())
    data_list = [Data(x=paddle.randn(shape=[x, 1])) for x in [2, 3, 10, 4]]
    batches = module.scatter(data_list, device_ids=[0, 1, 0, 1])
    assert len(batches) == 3

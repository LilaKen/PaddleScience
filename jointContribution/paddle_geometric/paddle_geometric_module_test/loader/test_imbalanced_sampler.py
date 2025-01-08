from typing import List

import paddle

from paddle_geometric.data import Data
from paddle_geometric.datasets import FakeDataset, FakeHeteroDataset
from paddle_geometric.loader import (
    DataLoader,
    ImbalancedSampler,
    NeighborLoader,
)
from paddle_geometric.testing import onlyNeighborSampler


def test_dataloader_with_imbalanced_sampler():
    data_list: List[Data] = []
    for _ in range(10):
        data_list.append(Data(num_nodes=10, y=0))
    for _ in range(90):
        data_list.append(Data(num_nodes=10, y=1))

    paddle.seed(12345)
    sampler = ImbalancedSampler(data_list)
    loader = DataLoader(data_list, batch_size=10, sampler=sampler)

    y = paddle.concat([batch.y for batch in loader])

    histogram = y.bincount()
    prob = histogram / histogram.sum()

    assert histogram.sum() == len(data_list)
    assert prob.min() > 0.4 and prob.max() < 0.6

    # Test with label tensor as input:
    paddle.seed(12345)
    sampler = ImbalancedSampler(paddle.to_tensor([data.y for data in data_list]))
    loader = DataLoader(data_list, batch_size=10, sampler=sampler)

    assert paddle.allclose(y, paddle.concat([batch.y for batch in loader]))

    # Test with list of data objects as input where each y is a tensor:
    paddle.seed(12345)
    for data in data_list:
        data.y = paddle.to_tensor([data.y])
    sampler = ImbalancedSampler(data_list)
    loader = DataLoader(data_list, batch_size=100, sampler=sampler)

    assert paddle.allclose(y, paddle.concat([batch.y for batch in loader]))


def test_in_memory_dataset_imbalanced_sampler():
    paddle.seed(12345)
    dataset = FakeDataset(num_graphs=100, avg_num_nodes=10, avg_degree=0,
                          num_channels=0, num_classes=2)
    sampler = ImbalancedSampler(dataset)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler)

    y = paddle.concat([batch.y for batch in loader])
    histogram = y.bincount()
    prob = histogram / histogram.sum()

    assert histogram.sum() == len(dataset)
    assert prob.min() > 0.4 and prob.max() < 0.6


@onlyNeighborSampler
def test_neighbor_loader_with_imbalanced_sampler():
    zeros = paddle.zeros(10, dtype=paddle.int64)
    ones = paddle.ones(90, dtype=paddle.int64)

    y = paddle.concat([zeros, ones], dim=0)
    edge_index = paddle.empty((2, 0), dtype=paddle.int64)
    data = Data(edge_index=edge_index, y=y, num_nodes=y.shape[0])

    paddle.seed(12345)
    sampler = ImbalancedSampler(data)
    loader = NeighborLoader(data, batch_size=10, sampler=sampler,
                            num_neighbors=[-1])

    y = paddle.concat([batch.y for batch in loader])

    histogram = y.bincount()
    prob = histogram / histogram.sum()

    assert histogram.sum() == data.num_nodes
    assert prob.min() > 0.4 and prob.max() < 0.6

    # Test with label tensor as input:
    paddle.seed(12345)
    sampler = ImbalancedSampler(data.y)
    loader = NeighborLoader(data, batch_size=10, sampler=sampler,
                            num_neighbors=[-1])

    assert paddle.allclose(y, paddle.concat([batch.y for batch in loader]))


@onlyNeighborSampler
def test_hetero_neighbor_loader_with_imbalanced_sampler():
    paddle.seed(12345)
    data = FakeHeteroDataset(num_classes=2)[0]

    loader = NeighborLoader(
        data,
        batch_size=100,
        input_nodes='v0',
        num_neighbors=[-1],
        sampler=ImbalancedSampler(data['v0'].y),
    )

    y = paddle.concat([batch['v0'].y[:batch['v0'].batch_size] for batch in loader])

    histogram = y.bincount()
    prob = histogram / histogram.sum()

    assert histogram.sum() == data['v0'].num_nodes
    assert prob.min() > 0.4 and prob.max() < 0.6

import pytest
import paddle
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.datasets import KarateClub
from paddle_geometric.loader.ibmb_loader import IBMBBatchLoader, IBMBNodeLoader
from paddle_geometric.testing import withPackage
from paddle_geometric.typing import SparseTensor


@withPackage('python_tsp')
@pytest.mark.parametrize(
    'use_sparse_tensor',
    [False] + [True] if paddle_geometric.typing.WITH_PADDLE_SPARSE else [])
@pytest.mark.parametrize('kwargs', [
    dict(num_partitions=4, batch_size=1),
    dict(num_partitions=8, batch_size=2),
])
def test_ibmb_batch_loader(use_sparse_tensor, kwargs):
    data = KarateClub()[0]

    loader = IBMBBatchLoader(
        data,
        batch_order='order',
        input_nodes=paddle.randperm(data.num_nodes)[:20],
        return_edge_index_type='adj' if use_sparse_tensor else 'edge_index',
        **kwargs,
    )
    assert str(loader) == 'IBMBBatchLoader()'
    assert len(loader) == 4
    assert sum([batch.output_node_mask.sum() for batch in loader]) == 20

    for batch in loader:
        if use_sparse_tensor:
            assert isinstance(batch.edge_index, SparseTensor)
        else:
            assert isinstance(batch.edge_index, Tensor)


@withPackage('python_tsp', 'numba')
@pytest.mark.parametrize(
    'use_sparse_tensor',
    [False] + [True] if paddle_geometric.typing.WITH_PADDLE_SPARSE else [])
@pytest.mark.parametrize('kwargs', [
    dict(num_nodes_per_batch=4, batch_size=1),
    dict(num_nodes_per_batch=2, batch_size=2),
])
def test_ibmb_node_loader(use_sparse_tensor, kwargs):
    data = KarateClub()[0]

    loader = IBMBNodeLoader(
        data,
        batch_order='order',
        input_nodes=paddle.randperm(data.num_nodes)[:20],
        num_auxiliary_nodes=4,
        return_edge_index_type='adj' if use_sparse_tensor else 'edge_index',
        **kwargs,
    )
    assert str(loader) == 'IBMBNodeLoader()'
    assert len(loader) == 5
    assert sum([batch.output_node_mask.sum() for batch in loader]) == 20

    for batch in loader:
        if use_sparse_tensor:
            assert isinstance(batch.edge_index, SparseTensor)
        else:
            assert isinstance(batch.edge_index, Tensor)

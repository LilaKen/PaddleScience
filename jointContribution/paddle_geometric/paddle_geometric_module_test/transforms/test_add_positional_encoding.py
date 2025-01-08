import paddle

from paddle_geometric.data import Data
from paddle_geometric.testing import withPackage
from paddle_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
)


@withPackage('scipy')
def test_add_laplacian_eigenvector_pe():
    x = paddle.randn(shape=[6, 4])
    edge_index = paddle.to_tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    transform = AddLaplacianEigenvectorPE(k=3)
    assert str(transform) == 'AddLaplacianEigenvectorPE()'
    out = transform(data)
    assert out.laplacian_eigenvector_pe.shape== (6, 3)

    transform = AddLaplacianEigenvectorPE(k=3, attr_name=None)
    out = transform(data)
    assert out.x.shape == [6, 4 + 3]

    transform = AddLaplacianEigenvectorPE(k=3, attr_name='x')
    out = transform(data)
    assert out.x.shape == [6, 3]

    # Output tests:
    edge_index = paddle.to_tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5, 2, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3, 5, 2]])
    data = Data(x=x, edge_index=edge_index)

    transform1 = AddLaplacianEigenvectorPE(k=1, is_undirected=True)
    transform2 = AddLaplacianEigenvectorPE(k=1, is_undirected=False)

    # Clustering test with first non-trivial eigenvector (Fiedler vector)
    pe = transform1(data).laplacian_eigenvector_pe
    pe_cluster_1 = pe[[0, 1, 4]]
    pe_cluster_2 = pe[[2, 3, 5]]
    assert not paddle.allclose(pe_cluster_1, pe_cluster_2)
    assert paddle.allclose(pe_cluster_1, pe_cluster_1.mean())
    assert paddle.allclose(pe_cluster_2, pe_cluster_2.mean())

    pe = transform2(data).laplacian_eigenvector_pe
    pe_cluster_1 = pe[[0, 1, 4]]
    pe_cluster_2 = pe[[2, 3, 5]]
    assert not paddle.allclose(pe_cluster_1, pe_cluster_2)
    assert paddle.allclose(pe_cluster_1, pe_cluster_1.mean())
    assert paddle.allclose(pe_cluster_2, pe_cluster_2.mean())


@withPackage('scipy')
def test_eigenvector_permutation_invariance():
    edge_index = paddle.to_tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(edge_index=edge_index, num_nodes=6)

    perm = paddle.randperm(data.num_nodes)
    transform = AddLaplacianEigenvectorPE(
        k=2,
        is_undirected=True,
        attr_name='x',
    )
    out1 = transform(data)

    transform = AddLaplacianEigenvectorPE(
        k=2,
        is_undirected=True,
        attr_name='x',
    )
    out2 = transform(data.subgraph(perm))

    assert paddle.allclose(out1.x[perm].abs(), out2.x.abs(), atol=1e-6)


def test_add_random_walk_pe():
    x = paddle.randn(shape=[6, 4])
    edge_index = paddle.to_tensor([[0, 1, 0, 4, 1, 4, 2, 3, 3, 5],
                               [1, 0, 4, 0, 4, 1, 3, 2, 5, 3]])
    data = Data(x=x, edge_index=edge_index)

    transform = AddRandomWalkPE(walk_length=3)
    assert str(transform) == 'AddRandomWalkPE()'
    out = transform(data)
    assert out.random_walk_pe.shape == [6, 3]

    transform = AddRandomWalkPE(walk_length=3, attr_name=None)
    out = transform(data)
    assert out.x.shape== [6, 4 + 3]

    transform = AddRandomWalkPE(walk_length=3, attr_name='x')
    out = transform(data)
    assert out.x.shape == [6, 3]

    # Output tests:
    assert out.x.tolist() == [
        [0.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 1.0, 0.00],
        [0.0, 2.0, 0.00],
        [0.0, 2.0, 2.0],
        [0.0, 1.0, 0.00],
    ]

    edge_index = paddle.to_tensor([[0, 1, 2], [0, 1, 2]])
    data = Data(edge_index=edge_index, num_nodes=4)
    out = transform(data)

    assert out.x.tolist() == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]

import paddle

from paddle_geometric.utils import (
    batched_negative_sampling,
    contains_self_loops,
    is_undirected,
    negative_sampling,
    structured_negative_sampling,
    structured_negative_sampling_feasible,
    to_undirected,
)
from paddle_geometric.utils._negative_sampling import (
    edge_index_to_vector,
    vector_to_edge_index,
)


def is_negative(edge_index, neg_edge_index, size, bipartite):
    adj = paddle.zeros(size, dtype=paddle.bool)
    neg_adj = paddle.zeros(size, dtype=paddle.bool)

    adj[edge_index[0], edge_index[1]] = True
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    if not bipartite:
        arange = paddle.arange(size[0])
        assert neg_adj[arange, arange].sum() == 0

    return (adj & neg_adj).sum() == 0


def test_edge_index_to_vector_and_vice_versa():
    # Create a fully-connected graph:
    N = 10
    row = paddle.arange(N).view(-1, 1).repeat(1, N).view(-1)
    col = paddle.arange(N).view(1, -1).repeat(N, 1).view(-1)
    edge_index = paddle.stack([row, col], dim=0)

    idx, population = edge_index_to_vector(edge_index, (N, N), bipartite=True)
    assert population == N * N
    assert idx.tolist() == list(range(population))
    edge_index2 = vector_to_edge_index(idx, (N, N), bipartite=True)
    assert is_undirected(edge_index2)
    assert edge_index.tolist() == edge_index2.tolist()

    idx, population = edge_index_to_vector(edge_index, (N, N), bipartite=False)
    assert population == N * N - N
    assert idx.tolist() == list(range(population))
    mask = edge_index[0] != edge_index[1]  # Remove self-loops.
    edge_index2 = vector_to_edge_index(idx, (N, N), bipartite=False)
    assert is_undirected(edge_index2)
    assert edge_index[:, mask].tolist() == edge_index2.tolist()

    idx, population = edge_index_to_vector(edge_index, (N, N), bipartite=False,
                                           force_undirected=True)
    assert population == (N * (N + 1)) / 2 - N
    assert idx.tolist() == list(range(population))
    mask = edge_index[0] != edge_index[1]  # Remove self-loops.
    edge_index2 = vector_to_edge_index(idx, (N, N), bipartite=False,
                                       force_undirected=True)
    assert is_undirected(edge_index2)
    assert edge_index[:, mask].tolist() == to_undirected(edge_index2).tolist()


def test_negative_sampling():
    edge_index = paddle.to_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_sampling(edge_index)
    assert neg_edge_index.shape[1] == edge_index.shape[1]
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    neg_edge_index = negative_sampling(edge_index, method='dense')
    assert neg_edge_index.shape[1] == edge_index.shape[1]
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    neg_edge_index = negative_sampling(edge_index, num_neg_samples=2)
    assert neg_edge_index.shape[1] == 2
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    edge_index = to_undirected(edge_index)
    neg_edge_index = negative_sampling(edge_index, force_undirected=True)
    assert neg_edge_index.shape[1] == edge_index.shape[1] - 1
    assert is_undirected(neg_edge_index)
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)


def test_bipartite_negative_sampling():
    edge_index = paddle.to_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_sampling(edge_index, num_nodes=(3, 4))
    assert neg_edge_index.shape[1] == edge_index.shape[1]
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)

    neg_edge_index = negative_sampling(edge_index, num_nodes=(3, 4),
                                       num_neg_samples=2)
    assert neg_edge_index.shape[1] == 2
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)


def test_batched_negative_sampling():
    edge_index = paddle.to_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    edge_index = paddle.concat([edge_index, edge_index + 4], dim=1)
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1])

    neg_edge_index = batched_negative_sampling(edge_index, batch)
    assert neg_edge_index.shape[1] <= edge_index.shape[1]

    adj = paddle.zeros(8, 8, dtype=paddle.bool)
    adj[edge_index[0], edge_index[1]] = True
    neg_adj = paddle.zeros(8, 8, dtype=paddle.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    assert (adj & neg_adj).sum() == 0
    assert (adj | neg_adj).sum() == edge_index.shape[1] + neg_edge_index.shape[1]
    assert neg_adj[:4, 4:].sum() == 0
    assert neg_adj[4:, :4].sum() == 0


def test_bipartite_batched_negative_sampling():
    edge_index1 = paddle.to_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
    edge_index2 = edge_index1 + paddle.to_tensor([[2], [4]])
    edge_index3 = edge_index2 + paddle.to_tensor([[2], [4]])
    edge_index = paddle.concat([edge_index1, edge_index2, edge_index3], dim=1)
    src_batch = paddle.to_tensor([0, 0, 1, 1, 2, 2])
    dst_batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    neg_edge_index = batched_negative_sampling(edge_index,
                                               (src_batch, dst_batch))
    assert neg_edge_index.shape[1] <= edge_index.shape[1]

    adj = paddle.zeros(6, 12, dtype=paddle.bool)
    adj[edge_index[0], edge_index[1]] = True
    neg_adj = paddle.zeros(6, 12, dtype=paddle.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    assert (adj & neg_adj).sum() == 0
    assert (adj | neg_adj).sum() == edge_index.shape[1] + neg_edge_index.shape[1]


def test_structured_negative_sampling():
    edge_index = paddle.to_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    i, j, k = structured_negative_sampling(edge_index)
    assert i.shape[0] == edge_index.shape[1]
    assert j.shape[0] == edge_index.shape[1]
    assert k.shape[0] == edge_index.shape[1]

    adj = paddle.zeros(4, 4, dtype=paddle.bool)
    adj[i, j] = 1

    neg_adj = paddle.zeros(4, 4, dtype=paddle.bool)
    neg_adj[i, k] = 1
    assert (adj & neg_adj).sum() == 0

    # Test with no self-loops:
    edge_index = paddle.to_tensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 1]], dtype='int64')
    i, j, k = structured_negative_sampling(edge_index, num_nodes=4,
                                           contains_neg_self_loops=False)
    neg_edge_index = paddle.vstack([i, k])
    assert not contains_self_loops(neg_edge_index)


def test_structured_negative_sampling_feasible():
    edge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 2, 2],
                                   [1, 2, 0, 2, 0, 1, 1]])
    assert not structured_negative_sampling_feasible(edge_index, 3, False)
    assert structured_negative_sampling_feasible(edge_index, 3, True)
    assert structured_negative_sampling_feasible(edge_index, 4, False)

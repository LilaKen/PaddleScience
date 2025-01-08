import paddle

from paddle_geometric.nn.models.graph_mixer import (
    LinkEncoder,
    NodeEncoder,
    get_latest_k_edge_attr,
)


def test_node_encoder():
    x = paddle.arange(4, dtype=paddle.float32).view(-1, 1)
    edge_index = paddle.to_tensor([[1, 2, 0, 0, 1, 3], [0, 0, 1, 2, 2, 2]])
    edge_time = paddle.to_tensor([0, 1, 1, 1, 2, 3])
    seed_time = paddle.to_tensor([2, 2, 2, 2])

    encoder = NodeEncoder(time_window=2)
    encoder.reset_parameters()
    assert str(encoder) == 'NodeEncoder(time_window=2)'

    out = encoder(x, edge_index, edge_time, seed_time)
    # Node 0 aggregates information from node 2 (excluding node 1).
    # Node 1 aggregates information from node 0.
    # Node 2 aggregates information from node 0 and node 1 (exluding node 3).
    # Node 3 aggregates no information.
    expected = paddle.to_tensor([
        [0 + 2],
        [1 + 0],
        [2 + 0.5 * (0 + 1)],
        [3],
    ])
    assert paddle.allclose(out, expected)


def test_link_encoder():
    num_nodes = 3
    num_edges = 6
    edge_attr = paddle.rand((num_edges, 10))
    edge_index = paddle.randint(low=0, high=num_nodes, size=(2, num_edges))
    edge_time = paddle.rand(num_edges)
    seed_time = paddle.ones(num_nodes)

    encoder = LinkEncoder(
        k=3,
        in_channels=edge_attr.shape[1],
        hidden_channels=7,
        out_channels=11,
        time_channels=13,
    )
    encoder.reset_parameters()
    assert str(encoder) == ('LinkEncoder(k=3, in_channels=10, '
                            'hidden_channels=7, out_channels=11, '
                            'time_channels=13, dropout=0.0)')

    out = encoder(edge_index, edge_attr, edge_time, seed_time)
    assert out.shape== (num_nodes, 11)


def test_latest_k_edge_attr():
    edge_index = paddle.to_tensor([[0, 0, 1, 1, 2, 2, 0], [0, 1, 0, 1, 0, 1, 2]])
    edge_time = paddle.to_tensor([3, 1, 2, 3, 1, 2, 3])
    edge_attr = paddle.to_tensor([1, -1, 3, 4, -1, 6, 7]).view(-1, 1)

    k = 2
    out = get_latest_k_edge_attr(k, edge_index, edge_attr, edge_time,
                                 num_nodes=3)
    expected = paddle.to_tensor([[[1], [3]], [[4], [6]], [[7], [0]]])
    assert out.shape== (3, 2, 1)
    assert paddle.equal(out, expected)

    k = 1
    out = get_latest_k_edge_attr(k, edge_index, edge_attr, edge_time,
                                 num_nodes=3)
    expected = paddle.to_tensor([[[1]], [[4]], [[7]]])
    assert out.shape== (3, 1, 1)
    assert paddle.equal(out, expected)

import io

import paddle

import paddle_geometric.typing
from paddle_geometric.nn import ASAPooling, GCNConv, GraphConv
from paddle_geometric.testing import is_full_test, onlyFullTest, onlyLinux


@onlyLinux  # TODO  (matthias) Investigate CSR @ CSR support on Windows.
def test_asap():
    in_channels = 16
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = paddle.randn(shape=[num_nodes, in_channels])

    for GNN in [GraphConv, GCNConv]:
        pool = ASAPooling(in_channels, ratio=0.5, GNN=GNN,
                          add_self_loops=False)
        assert str(pool) == ('ASAPooling(16, ratio=0.5)')
        out = pool(x, edge_index)
        assert out[0].shape== (num_nodes // 2, in_channels)
        assert out[1].shape== (2, 2)

        if is_full_test():
            paddle.jit.to_static(pool)

        pool = ASAPooling(in_channels, ratio=0.5, GNN=GNN, add_self_loops=True)
        assert str(pool) == ('ASAPooling(16, ratio=0.5)')
        out = pool(x, edge_index)
        assert out[0].shape== (num_nodes // 2, in_channels)
        assert out[1].shape== (2, 4)

        pool = ASAPooling(in_channels, ratio=2, GNN=GNN, add_self_loops=False)
        assert str(pool) == ('ASAPooling(16, ratio=2)')
        out = pool(x, edge_index)
        assert out[0].shape== (2, in_channels)
        assert out[1].shape== (2, 2)


@onlyFullTest
def test_asap_jit_save():
    pool = ASAPooling(in_channels=16)
    paddle.jit.save(paddle.jit.to_static(pool), io.BytesIO())

import paddle

import paddle_geometric.typing
from paddle_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    SAGEConv,
    SAGPooling,
)
from paddle_geometric.testing import is_full_test


def test_sag_pooling():
    in_channels = 16
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])
    num_nodes = edge_index.max().item() + 1
    x = paddle.randn(shape=[(num_nodes, in_channels]))

    for GNN in [GraphConv, GCNConv, GATConv, SAGEConv]:
        pool1 = SAGPooling(in_channels, ratio=0.5, GNN=GNN)
        assert str(pool1) == (f'SAGPooling({GNN.__name__}, 16, '
                              f'ratio=0.5, multiplier=1.0)')
        out1 = pool1(x, edge_index)
        assert out1[0].shape== (num_nodes // 2, in_channels)
        assert out1[1].shape== (2, 2)

        pool2 = SAGPooling(in_channels, ratio=None, GNN=GNN, min_score=0.1)
        assert str(pool2) == (f'SAGPooling({GNN.__name__}, 16, '
                              f'min_score=0.1, multiplier=1.0)')
        out2 = pool2(x, edge_index)
        assert out2[0].shape[0] <= x.shape[0] and out2[0].shape[1] == (16)
        assert out2[1].shape[0] == 2 and out2[1].shape[1] <= edge_index.shape[1]

        pool3 = SAGPooling(in_channels, ratio=2, GNN=GNN)
        assert str(pool3) == (f'SAGPooling({GNN.__name__}, 16, '
                              f'ratio=2, multiplier=1.0)')
        out3 = pool3(x, edge_index)
        assert out3[0].shape== (2, in_channels)
        assert out3[1].shape== (2, 2)

        if is_full_test():
            jit1 = paddle.jit.to_static(pool1)
            assert paddle.allclose(jit1(x, edge_index)[0], out1[0])

            jit2 = paddle.jit.to_static(pool2)
            assert paddle.allclose(jit2(x, edge_index)[0], out2[0])

            jit3 = paddle.jit.to_static(pool3)
            assert paddle.allclose(jit3(x, edge_index)[0], out3[0])

import paddle

import paddle_geometric.typing
from paddle_geometric.nn import HANConv
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import coalesce, to_paddle_csc_tensor


def test_han_conv():
    x_dict = {
        'author': paddle.randn(shape=[6, 16]),
        'paper': paddle.randn(shape=[5, 12]),
        'term': paddle.randn(shape=[4, 3])
    }
    edge_index1 = coalesce(paddle.randint(0, 6, (2, 7)))
    edge_index2 = coalesce(paddle.randint(0, 5, (2, 4)))
    edge_index3 = coalesce(paddle.randint(0, 3, (2, 5)))
    edge_index_dict = {
        ('author', 'metapath0', 'author'): edge_index1,
        ('paper', 'metapath1', 'paper'): edge_index2,
        ('paper', 'metapath2', 'paper'): edge_index3,
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_paddle_csc_tensor(
            edge_index,
            size=(x_dict[src_type].shape[0], x_dict[dst_type].shape[0]),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12, 'term': 3}

    conv = HANConv(in_channels, 16, metadata, heads=2)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 3
    assert out_dict1['author'].shape== (6, 16)
    assert out_dict1['paper'].shape== (5, 16)
    assert out_dict1['term'] is None
    del out_dict1['term']
    del x_dict['term']

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert paddle.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for key in out_dict3.keys():
            assert paddle.allclose(out_dict1[key], out_dict3[key], atol=1e-6)

    # Test non-zero dropout:
    conv = HANConv(in_channels, 16, metadata, heads=2, dropout=0.1)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].shape== (6, 16)
    assert out_dict1['paper'].shape== (5, 16)


def test_han_conv_lazy():
    x_dict = {
        'author': paddle.randn(shape=[6, 16]),
        'paper': paddle.randn(shape=[5, 12]),
    }
    edge_index1 = coalesce(paddle.randint(0, 6, (2, 8)))
    edge_index2 = coalesce(paddle.randint(0, 5, (2, 6)))
    edge_index_dict = {
        ('author', 'to', 'author'): edge_index1,
        ('paper', 'to', 'paper'): edge_index2,
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_paddle_csc_tensor(
            edge_index,
            size=(x_dict[src_type].shape[0], x_dict[dst_type].shape[0]),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    conv = HANConv(-1, 16, metadata, heads=2)
    assert str(conv) == 'HANConv(16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].shape== (6, 16)
    assert out_dict1['paper'].shape== (5, 16)

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert paddle.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for key in out_dict1.keys():
            assert paddle.allclose(out_dict1[key], out_dict3[key], atol=1e-6)


def test_han_conv_empty_tensor():
    x_dict = {
        'author': paddle.randn(shape=[6, 16]),
        'paper': paddle.empty(0, 12),
    }
    edge_index_dict = {
        ('paper', 'to', 'author'): paddle.empty((2, 0), dtype=paddle.long),
        ('author', 'to', 'paper'): paddle.empty((2, 0), dtype=paddle.long),
        ('paper', 'to', 'paper'): paddle.empty((2, 0), dtype=paddle.long),
    }

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12}
    conv = HANConv(in_channels, 16, metadata, heads=2)

    out_dict = conv(x_dict, edge_index_dict)
    assert len(out_dict) == 2
    assert out_dict['author'].shape== (6, 16)
    assert paddle.all(out_dict['author'] == 0)
    assert out_dict['paper'].shape== (0, 16)

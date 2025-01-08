import paddle

import paddle_geometric.typing
from paddle_geometric.nn.pool.connect import FilterEdges
from paddle_geometric.nn.pool.select import SelectOutput
from paddle_geometric.testing import is_full_test


def test_filter_edges():
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 1, 3, 2, 2]])
    edge_attr = paddle.to_tensor([1, 2, 3, 4, 5, 6])
    batch = paddle.to_tensor([0, 0, 1, 1])

    select_output = SelectOutput(
        node_index=paddle.to_tensor([1, 2]),
        num_nodes=4,
        cluster_index=paddle.to_tensor([0, 1]),
        num_clusters=2,
    )

    connect = FilterEdges()
    assert str(connect) == 'FilterEdges()'

    out1 = connect(select_output, edge_index, edge_attr, batch)
    assert out1.edge_index.tolist() == [[0, 1], [0, 1]]
    assert out1.edge_attr.tolist() == [3, 5]
    assert out1.batch.tolist() == [0, 1]

    if is_full_test():
        jit = paddle.jit.to_static(connect)
        out2 = jit(select_output, edge_index, edge_attr, batch)
        paddle.equal(out1.edge_index, out2.edge_index)
        paddle.equal(out1.edge_attr, out2.edge_attr)
        paddle.equal(out1.batch, out2.batch)

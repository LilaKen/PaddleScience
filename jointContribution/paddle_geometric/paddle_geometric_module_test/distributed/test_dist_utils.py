import paddle

from paddle_geometric.distributed.utils import remove_duplicates
from paddle_geometric.sampler import SamplerOutput
from paddle_geometric.testing import onlyDistributedTest


@onlyDistributedTest
def test_remove_duplicates():
    node = paddle.to_tensor([0, 1, 2, 3])
    out_node = paddle.to_tensor([0, 4, 1, 5, 1, 6, 2, 7, 3, 8])

    out = SamplerOutput(out_node, None, None, None)

    src, node, _, _ = remove_duplicates(out, node)

    assert src.tolist() == [4, 5, 6, 7, 8]
    assert node.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]


@onlyDistributedTest
def test_remove_duplicates_disjoint():
    node = paddle.to_tensor([0, 1, 2, 3])
    batch = paddle.to_tensor([0, 1, 2, 3])

    out_node = paddle.to_tensor([0, 4, 1, 5, 1, 6, 2, 6, 7, 3, 8])
    out_batch = paddle.to_tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    out = SamplerOutput(out_node, None, None, None, out_batch)

    src, node, src_batch, batch = remove_duplicates(out, node, batch,
                                                    disjoint=True)

    assert src.tolist() == [4, 5, 6, 7, 8]
    assert node.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    assert src_batch.tolist() == [0, 1, 2, 3, 3]
    assert batch.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 3]

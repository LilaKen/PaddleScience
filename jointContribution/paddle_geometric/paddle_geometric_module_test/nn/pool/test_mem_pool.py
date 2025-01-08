import paddle

from paddle_geometric.nn import MemPooling
from paddle_geometric.utils import to_dense_batch


def test_mem_pool():
    mpool1 = MemPooling(4, 8, heads=3, num_clusters=2)
    assert str(mpool1) == 'MemPooling(4, 8, heads=3, num_clusters=2)'
    mpool2 = MemPooling(8, 4, heads=2, num_clusters=1)

    x = paddle.randn(shape=[17, 4])
    batch = paddle.to_tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    _, mask = to_dense_batch(x, batch)

    out1, S = mpool1(x, batch)
    loss = MemPooling.kl_loss(S)
    with paddle.autograd.set_detect_anomaly(True):
        loss.backward()
    out2, _ = mpool2(out1)

    assert out1.shape== (5, 2, 8)
    assert out2.shape== (5, 1, 4)
    assert S[~mask].sum() == 0
    assert round(S[mask].sum().item()) == x.shape[0]
    assert float(loss) > 0

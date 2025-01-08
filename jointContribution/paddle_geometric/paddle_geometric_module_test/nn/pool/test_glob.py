import paddle

from paddle_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


def test_global_pool():
    N_1, N_2 = 4, 6
    x = paddle.randn(shape=[N_1 + N_2, 4])
    batch = paddle.to_tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])

    out = global_add_pool(x, batch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out[0], x[:4].sum(dim=0))
    assert paddle.allclose(out[1], x[4:].sum(dim=0))

    out = global_add_pool(x, None)
    assert out.shape== (1, 4)
    assert paddle.allclose(out, x.sum(dim=0, keepdim=True))

    out = global_mean_pool(x, batch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out[0], x[:4].mean(dim=0))
    assert paddle.allclose(out[1], x[4:].mean(dim=0))

    out = global_mean_pool(x, None)
    assert out.shape== (1, 4)
    assert paddle.allclose(out, x.mean(dim=0, keepdim=True))

    out = global_max_pool(x, batch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out[0], x[:4].max(dim=0)[0])
    assert paddle.allclose(out[1], x[4:].max(dim=0)[0])

    out = global_max_pool(x, None)
    assert out.shape== (1, 4)
    assert paddle.allclose(out, x.max(dim=0, keepdim=True)[0])


def test_permuted_global_pool():
    N_1, N_2 = 4, 6
    x = paddle.randn(shape=[N_1 + N_2, 4])
    batch = paddle.concat([paddle.zeros(N_1), paddle.ones(N_2)]).to(paddle.int64)
    perm = paddle.randperm(N_1 + N_2)

    px = x[perm]
    pbatch = batch[perm]
    px1 = px[pbatch == 0]
    px2 = px[pbatch == 1]

    out = global_add_pool(px, pbatch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out[0], px1.sum(dim=0))
    assert paddle.allclose(out[1], px2.sum(dim=0))

    out = global_mean_pool(px, pbatch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out[0], px1.mean(dim=0))
    assert paddle.allclose(out[1], px2.mean(dim=0))

    out = global_max_pool(px, pbatch)
    assert out.shape== (2, 4)
    assert paddle.allclose(out[0], px1.max(dim=0)[0])
    assert paddle.allclose(out[1], px2.max(dim=0)[0])


def test_dense_global_pool():
    x = paddle.randn(shape=[3, 16, 32])
    assert paddle.allclose(global_add_pool(x, None), x.sum(dim=1))

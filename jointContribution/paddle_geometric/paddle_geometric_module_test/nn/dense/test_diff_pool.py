from itertools import product

import paddle

from paddle_geometric.nn import dense_diff_pool
from paddle_geometric.profile import benchmark
from paddle_geometric.testing import is_full_test


def test_dense_diff_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = paddle.randn(shape=[(batch_size, num_nodes, channels]))
    adj = paddle.rand((batch_size, num_nodes, num_nodes))
    s = paddle.randn(shape=[(batch_size, num_nodes, num_clusters]))
    mask = paddle.randint(0, 2, (batch_size, num_nodes), dtype=paddle.bool)

    x_out, adj_out, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
    assert x_out.shape== (2, 10, 16)
    assert adj_out.shape== (2, 10, 10)
    assert link_loss.item() >= 0
    assert ent_loss.item() >= 0

    if is_full_test():
        jit = paddle.jit.to_static(dense_diff_pool)
        x_jit, adj_jit, link_loss, ent_loss = jit(x, adj, s, mask)
        assert paddle.allclose(x_jit, x_out)
        assert paddle.allclose(adj_jit, adj_out)
        assert link_loss.item() >= 0
        assert ent_loss.item() >= 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    BS = [2**i for i in range(4, 8)]
    NS = [2**i for i in range(4, 8)]
    FS = [2**i for i in range(5, 9)]
    CS = [2**i for i in range(5, 9)]

    funcs = []
    func_names = []
    args_list = []
    for B, N, F, C in product(BS, NS, FS, CS):
        x = paddle.randn(shape=[B, N, F, place=args.device])
        adj = paddle.randint(0, 2, (B, N, N), dtype=x.dtype, place=args.device)
        s = paddle.randn(shape=[B, N, C, place=args.device])

        funcs.append(dense_diff_pool)
        func_names.append(f'B={B}, N={N}, F={F}, C={C}')
        args_list.append((x, adj, s))

    benchmark(
        funcs=funcs,
        func_names=func_names,
        args=args_list,
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
        progress_bar=True,
    )

import paddle

from paddle_geometric.nn.functional import bro


def test_bro():
    batch = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])

    g1 = paddle.to_tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
        [0.2, 0.2, 0.0, 0.2],
    ])

    g2 = paddle.to_tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ])

    g3 = paddle.to_tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ])

    s = 0.
    for g in [g1, g2, g3]:
        s += paddle.norm(g @ g.t() - paddle.eye(g.shape[0]), p=2)

    assert paddle.isclose(s / 3., bro(paddle.concat([g1, g2, g3], dim=0), batch))

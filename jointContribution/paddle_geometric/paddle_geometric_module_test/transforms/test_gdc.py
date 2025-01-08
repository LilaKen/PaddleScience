import paddle

from paddle_geometric.datasets import KarateClub
from paddle_geometric.testing import withPackage
from paddle_geometric.transforms import GDC
from paddle_geometric.utils import to_dense_adj


@withPackage('numba')
def test_gdc():
    data = KarateClub()[0]

    gdc = GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='sym',
        diffusion_kwargs=dict(method='ppr', alpha=0.15),
        sparsification_kwargs=dict(method='threshold', avg_degree=2),
        exact=True,
    )
    out = gdc(data)
    mat = to_dense_adj(out.edge_index, edge_attr=out.edge_attr).squeeze()
    assert paddle.all(mat >= -1e-8)
    assert paddle.allclose(mat, mat.t(), atol=1e-4)

    gdc = GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='sym',
        diffusion_kwargs=dict(method='heat', t=10),
        sparsification_kwargs=dict(method='threshold', avg_degree=2),
        exact=True,
    )
    out = gdc(data)
    mat = to_dense_adj(out.edge_index, edge_attr=out.edge_attr).squeeze()
    assert paddle.all(mat >= -1e-8)
    assert paddle.allclose(mat, mat.t(), atol=1e-4)

    gdc = GDC(
        self_loop_weight=1,
        normalization_in='col',
        normalization_out='col',
        diffusion_kwargs=dict(method='heat', t=10),
        sparsification_kwargs=dict(method='topk', k=2, dim=0),
        exact=True,
    )
    out = gdc(data)
    mat = to_dense_adj(out.edge_index, edge_attr=out.edge_attr).squeeze()
    col_sum = mat.sum(0)
    assert paddle.all(mat >= -1e-8)
    assert paddle.all(
        paddle.isclose(col_sum, paddle.to_tensor(1.0))
        | paddle.isclose(col_sum, paddle.to_tensor(0.0)))
    assert paddle.all((~paddle.isclose(mat, paddle.to_tensor(0.0))).sum(0) == 2)

    gdc = GDC(
        self_loop_weight=1,
        normalization_in='row',
        normalization_out='row',
        diffusion_kwargs=dict(method='heat', t=5),
        sparsification_kwargs=dict(method='topk', k=2, dim=1),
        exact=True,
    )
    out = gdc(data)
    mat = to_dense_adj(out.edge_index, edge_attr=out.edge_attr).squeeze()
    row_sum = mat.sum(1)
    assert paddle.all(mat >= -1e-8)
    assert paddle.all(
        paddle.isclose(row_sum, paddle.to_tensor(1.0))
        | paddle.isclose(row_sum, paddle.to_tensor(0.0)))
    assert paddle.all((~paddle.isclose(mat, paddle.to_tensor(0.0))).sum(1) == 2)

    gdc = GDC(
        self_loop_weight=1,
        normalization_in='row',
        normalization_out='row',
        diffusion_kwargs=dict(method='coeff', coeffs=[0.8, 0.3, 0.1]),
        sparsification_kwargs=dict(method='threshold', eps=0.1),
        exact=True,
    )
    out = gdc(data)
    mat = to_dense_adj(out.edge_index, edge_attr=out.edge_attr).squeeze()
    row_sum = mat.sum(1)
    assert paddle.all(mat >= -1e-8)
    assert paddle.all(
        paddle.isclose(row_sum, paddle.to_tensor(1.0))
        | paddle.isclose(row_sum, paddle.to_tensor(0.0)))

    gdc = GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.15, eps=1e-4),
        sparsification_kwargs=dict(method='threshold', avg_degree=2),
        exact=False,
    )
    out = gdc(data)
    mat = to_dense_adj(out.edge_index, edge_attr=out.edge_attr).squeeze()
    col_sum = mat.sum(0)
    assert paddle.all(mat >= -1e-8)
    assert paddle.all(
        paddle.isclose(col_sum, paddle.to_tensor(1.0))
        | paddle.isclose(col_sum, paddle.to_tensor(0.0)))

import pytest
import paddle

from paddle_geometric.nn import (
    ApproxL2KNNIndex,
    ApproxMIPSKNNIndex,
    L2KNNIndex,
    MIPSKNNIndex,
)
from paddle_geometric.testing import withCUDA, withPackage


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [2])
def test_l2(device, k):
    lhs = paddle.randn(shape=[10, 16, place=device])
    rhs = paddle.randn(shape=[100, 16, place=device])

    index = L2KNNIndex(rhs)
    assert index.get_emb().device == device
    assert paddle.equal(index.get_emb(), rhs)

    out = index.search(lhs, k)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.shape== (10, k)
    assert out.index.shape== (10, k)

    mat = paddle.linalg.norm(lhs.unsqueeze(1) - rhs.unsqueeze(0), dim=-1).pow(2)
    score, index = mat.sort(dim=-1)

    assert paddle.allclose(out.score, score[:, :k])
    assert paddle.equal(out.index, index[:, :k])


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [2])
def test_mips(device, k):
    lhs = paddle.randn(shape=[10, 16, place=device])
    rhs = paddle.randn(shape=[100, 16, place=device])

    index = MIPSKNNIndex(rhs)
    assert index.get_emb().device == device
    assert paddle.equal(index.get_emb(), rhs)

    out = index.search(lhs, k)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.shape== (10, k)
    assert out.index.shape== (10, k)

    mat = lhs @ rhs.t()
    score, index = mat.sort(dim=-1, descending=True)

    assert paddle.allclose(out.score, score[:, :k])
    assert paddle.equal(out.index, index[:, :k])


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [2])
@pytest.mark.parametrize('reserve', [None, 100])
def test_approx_l2(device, k, reserve):
    lhs = paddle.randn(shape=[10, 16, place=device])
    rhs = paddle.randn(shape=[10_000, 16, place=device])

    index = ApproxL2KNNIndex(
        num_cells=10,
        num_cells_to_visit=10,
        bits_per_vector=8,
        emb=rhs,
        reserve=reserve,
    )

    out = index.search(lhs, k)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.shape== (10, k)
    assert out.index.shape== (10, k)
    assert out.index.min() >= 0 and out.index.max() < 10_000


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [2])
@pytest.mark.parametrize('reserve', [None, 100])
def test_approx_mips(device, k, reserve):
    lhs = paddle.randn(shape=[10, 16, place=device])
    rhs = paddle.randn(shape=[10_000, 16, place=device])

    index = ApproxMIPSKNNIndex(
        num_cells=10,
        num_cells_to_visit=10,
        bits_per_vector=8,
        emb=rhs,
        reserve=reserve,
    )

    out = index.search(lhs, k)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.shape== (10, k)
    assert out.index.shape== (10, k)
    assert out.index.min() >= 0 and out.index.max() < 10_000


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [50])
def test_mips_exclude(device, k):
    lhs = paddle.randn(shape=[10, 16, place=device])
    rhs = paddle.randn(shape=[100, 16, place=device])

    exclude_lhs = paddle.randint(0, 10, (500, ), place=device)
    exclude_rhs = paddle.randint(0, 100, (500, ), place=device)
    exclude_links = paddle.stack([exclude_lhs, exclude_rhs], dim=0)
    exclude_links = exclude_links.unique(dim=1)

    index = MIPSKNNIndex(rhs)

    out = index.search(lhs, k, exclude_links)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.shape== (10, k)
    assert out.index.shape== (10, k)

    # Ensure that excluded links are not present in `out.index`:
    batch = paddle.arange(lhs.shape[0], place=device).repeat_interleave(k)
    knn_links = paddle.stack([batch, out.index.view(-1)], dim=0)
    knn_links = knn_links[:, knn_links[1] >= 0]

    unique_links = paddle.concat([knn_links, exclude_links], dim=1).unique(dim=1)
    assert unique_links.shape[1] == knn_links.shape[1] + exclude_links.shape[1]

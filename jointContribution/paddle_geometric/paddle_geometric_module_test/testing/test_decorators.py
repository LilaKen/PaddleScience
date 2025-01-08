import paddle_geometric.typing
from paddle_geometric.testing import disableExtensions


def test_enable_extensions():
    try:
        import pyg_lib  # noqa
        assert paddle_geometric.typing.WITH_PYG_LIB
    except (ImportError, OSError):
        assert not paddle_geometric.typing.WITH_PYG_LIB

    try:
        import paddle_scatter  # noqa
        assert paddle_geometric.typing.WITH_PADDLE_SCATTER
    except (ImportError, OSError):
        assert not paddle_geometric.typing.WITH_PADDLE_SCATTER

    try:
        import paddle_sparse  # noqa
        assert paddle_geometric.typing.WITH_PADDLE_SPARSE
    except (ImportError, OSError):
        assert not paddle_geometric.typing.WITH_PADDLE_SPARSE


@disableExtensions
def test_disable_extensions():
    assert not paddle_geometric.typing.WITH_PYG_LIB
    assert not paddle_geometric.typing.WITH_PADDLE_SCATTER
    assert not paddle_geometric.typing.WITH_PADDLE_SPARSE

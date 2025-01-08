import os.path as osp
from typing import Optional, Union

import pytest
import paddle

from paddle_geometric.data import HeteroData
from paddle_geometric.explain import HeteroExplanation
from paddle_geometric.explain.config import MaskType
from paddle_geometric.testing import withPackage


def create_random_explanation(
    hetero_data: HeteroData,
    node_mask_type: Optional[Union[MaskType, str]] = None,
    edge_mask_type: Optional[Union[MaskType, str]] = None,
):
    if node_mask_type is not None:
        node_mask_type = MaskType(node_mask_type)
    if edge_mask_type is not None:
        edge_mask_type = MaskType(edge_mask_type)

    out = HeteroExplanation()

    for key in ['paper', 'author']:
        out[key].x = hetero_data[key].x
        if node_mask_type == MaskType.object:
            out[key].node_mask = paddle.rand(hetero_data[key].num_nodes, 1)
        elif node_mask_type == MaskType.common_attributes:
            out[key].node_mask = paddle.rand(1, hetero_data[key].num_features)
        elif node_mask_type == MaskType.attributes:
            out[key].node_mask = paddle.rand_like(hetero_data[key].x)

    for key in [('paper', 'paper'), ('paper', 'author')]:
        out[key].edge_index = hetero_data[key].edge_index
        out[key].edge_attr = hetero_data[key].edge_attr
        if edge_mask_type == MaskType.object:
            out[key].edge_mask = paddle.rand(hetero_data[key].num_edges)

    return out


@pytest.mark.parametrize('node_mask_type',
                         [None, 'object', 'common_attributes', 'attributes'])
@pytest.mark.parametrize('edge_mask_type', [None, 'object'])
def test_available_explanations(hetero_data, node_mask_type, edge_mask_type):
    expected = []
    if node_mask_type:
        expected.append('node_mask')
    if edge_mask_type:
        expected.append('edge_mask')

    explanation = create_random_explanation(
        hetero_data,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )

    assert set(explanation.available_explanations) == set(expected)


def test_validate_explanation(hetero_data):
    explanation = create_random_explanation(hetero_data)
    explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 8 nodes"):
        explanation = create_random_explanation(hetero_data)
        explanation['paper'].node_mask = paddle.rand(5, 5)
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 5 features"):
        explanation = create_random_explanation(hetero_data, 'attributes')
        explanation['paper'].x = paddle.randn(shape=[8, 5])
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 10 edges"):
        explanation = create_random_explanation(hetero_data)
        explanation['paper', 'paper'].edge_mask = paddle.randn(shape=[5])
        explanation.validate(raise_on_error=True)


def test_node_mask():
    explanation = HeteroExplanation()
    explanation['paper'].node_mask = paddle.to_tensor([[1.], [0.], [1.], [1.]])
    explanation['author'].node_mask = paddle.to_tensor([[1.], [0.], [1.], [1.]])
    with pytest.warns(UserWarning, match="are isolated"):
        explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out['paper'].node_mask.shape== (3, 1)
    assert out['author'].node_mask.shape== (3, 1)

    out = explanation.get_complement_subgraph()
    assert out['paper'].node_mask.shape== (1, 1)
    assert out['author'].node_mask.shape== (1, 1)


def test_edge_mask():
    explanation = HeteroExplanation()
    explanation['paper'].num_nodes = 4
    explanation['author'].num_nodes = 4
    explanation['paper', 'author'].edge_index = paddle.to_tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    explanation['paper', 'author'].edge_mask = paddle.to_tensor([1., 0., 1., 1.])

    out = explanation.get_explanation_subgraph()
    assert out['paper'].num_nodes == 4
    assert out['author'].num_nodes == 4
    assert out['paper', 'author'].edge_mask.shape== (3, )
    assert paddle.equal(out['paper', 'author'].edge_index,
                       paddle.to_tensor([[0, 2, 3], [0, 2, 3]]))

    out = explanation.get_complement_subgraph()
    assert out['paper'].num_nodes == 4
    assert out['author'].num_nodes == 4
    assert out['paper', 'author'].edge_mask.shape== (1, )
    assert paddle.equal(out['paper', 'author'].edge_index,
                       paddle.to_tensor([[1], [1]]))


@withPackage('matplotlib')
@pytest.mark.parametrize('top_k', [2, None])
@pytest.mark.parametrize('node_mask_type', [None, 'attributes'])
def test_visualize_feature_importance(
    top_k,
    node_mask_type,
    tmp_path,
    hetero_data,
):
    explanation = create_random_explanation(
        hetero_data,
        node_mask_type=node_mask_type,
    )

    path = osp.join(tmp_path, 'feature_importance.png')

    if node_mask_type is None:
        with pytest.raises(KeyError, match="Tried to collect 'node_mask'"):
            explanation.visualize_feature_importance(path, top_k=top_k)
    else:
        explanation.visualize_feature_importance(path, top_k=top_k)
        assert osp.exists(path)

import os.path as osp
from typing import Optional, Union

import pytest
import paddle

from paddle_geometric.data import Data
from paddle_geometric.explain import Explanation
from paddle_geometric.explain.config import MaskType
from paddle_geometric.testing import withPackage


def create_random_explanation(
    data: Data,
    node_mask_type: Optional[Union[MaskType, str]] = None,
    edge_mask_type: Optional[Union[MaskType, str]] = None,
):
    if node_mask_type is not None:
        node_mask_type = MaskType(node_mask_type)
    if edge_mask_type is not None:
        edge_mask_type = MaskType(edge_mask_type)

    if node_mask_type == MaskType.object:
        node_mask = paddle.rand(data.x.shape[0], 1)
    elif node_mask_type == MaskType.common_attributes:
        node_mask = paddle.rand(1, data.x.shape[1])
    elif node_mask_type == MaskType.attributes:
        node_mask = paddle.rand_like(data.x)
    else:
        node_mask = None

    if edge_mask_type == MaskType.object:
        edge_mask = paddle.rand(data.edge_index.shape[1])
    else:
        edge_mask = None

    return Explanation(  # Create explanation.
        node_mask=node_mask,
        edge_mask=edge_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


@pytest.mark.parametrize('node_mask_type',
                         [None, 'object', 'common_attributes', 'attributes'])
@pytest.mark.parametrize('edge_mask_type', [None, 'object'])
def test_available_explanations(data, node_mask_type, edge_mask_type):
    expected = []
    if node_mask_type is not None:
        expected.append('node_mask')
    if edge_mask_type is not None:
        expected.append('edge_mask')

    explanation = create_random_explanation(
        data,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )

    assert set(explanation.available_explanations) == set(expected)


def test_validate_explanation(data):
    explanation = create_random_explanation(data)
    explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 5 nodes"):
        explanation = create_random_explanation(data, node_mask_type='object')
        explanation.x = paddle.randn(shape=[5, 5])
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 4 features"):
        explanation = create_random_explanation(data, 'attributes')
        explanation.x = paddle.randn(shape=[4, 4])
        explanation.validate(raise_on_error=True)

    with pytest.raises(ValueError, match="with 7 edges"):
        explanation = create_random_explanation(data, edge_mask_type='object')
        explanation.edge_index = paddle.randint(0, 4, (2, 7))
        explanation.validate(raise_on_error=True)


def test_node_mask(data):
    node_mask = paddle.to_tensor([[1.], [0.], [1.], [0.]])

    explanation = Explanation(
        node_mask=node_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )
    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out.node_mask.shape== (2, 1)
    assert (out.node_mask > 0.0).sum() == 2
    assert out.x.shape== (2, 3)
    assert out.edge_index.shape[1] <= 6
    assert out.edge_index.shape[1] == out.edge_attr.shape[0]

    out = explanation.get_complement_subgraph()
    assert out.node_mask.shape== (2, 1)
    assert (out.node_mask == 0.0).sum() == 2
    assert out.x.shape== (2, 3)
    assert out.edge_index.shape[1] <= 6
    assert out.edge_index.shape[1] == out.edge_attr.shape[0]


def test_edge_mask(data):
    edge_mask = paddle.to_tensor([1., 0., 1., 0., 0., 1.])

    explanation = Explanation(
        edge_mask=edge_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )
    explanation.validate(raise_on_error=True)

    out = explanation.get_explanation_subgraph()
    assert out.x.shape== (4, 3)
    assert out.edge_mask.shape== (3, )
    assert (out.edge_mask > 0.0).sum() == 3
    assert out.edge_index.shape== (2, 3)
    assert out.edge_attr.shape== (3, 3)

    out = explanation.get_complement_subgraph()
    assert out.x.shape== (4, 3)
    assert out.edge_mask.shape== (3, )
    assert (out.edge_mask == 0.0).sum() == 3
    assert out.edge_index.shape== (2, 3)
    assert out.edge_attr.shape== (3, 3)


@withPackage('matplotlib', 'pandas')
@pytest.mark.parametrize('top_k', [2, None])
@pytest.mark.parametrize('node_mask_type', [None, 'attributes'])
def test_visualize_feature_importance(tmp_path, data, top_k, node_mask_type):
    explanation = create_random_explanation(data, node_mask_type)

    path = osp.join(tmp_path, 'feature_importance.png')

    if node_mask_type is None:
        with pytest.raises(ValueError, match="node_mask' is not"):
            explanation.visualize_feature_importance(path, top_k=top_k)
    else:
        explanation.visualize_feature_importance(path, top_k=top_k)
        assert osp.exists(path)

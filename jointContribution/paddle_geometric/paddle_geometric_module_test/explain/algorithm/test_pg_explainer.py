import pytest
import paddle

from paddle_geometric.explain import Explainer, PGExplainer
from paddle_geometric.explain.config import (
    ModelConfig,
    ModelMode,
    ModelTaskLevel,
)
from paddle_geometric.nn import GCNConv, global_add_pool
from paddle_geometric.testing import withCUDA


class GCN(paddle.nn.Layer):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        if model_config.mode == ModelMode.multiclass_classification:
            out_channels = 7
        else:
            out_channels = 1

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, batch=None, edge_label_index=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        if self.model_config.task_level == ModelTaskLevel.graph:
            x = global_add_pool(x, batch)
        return x


@withCUDA
@pytest.mark.parametrize('mode', [
    ModelMode.binary_classification,
    ModelMode.multiclass_classification,
    ModelMode.regression,
])
def test_pg_explainer_node(device, check_explanation, mode):
    x = paddle.randn(shape=[8, 3])
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], place=device)

    if mode == ModelMode.binary_classification:
        target = paddle.randint(2, (x.shape[0], ), place=device)
    elif mode == ModelMode.multiclass_classification:
        target = paddle.randint(7, (x.shape[0], ), place=device)
    elif mode == ModelMode.regression:
        target = paddle.randn(shape=[x.shape[0], 1])

    model_config = ModelConfig(mode=mode, task_level='node', return_type='raw')

    model = GCN(model_config).to(device)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="not yet fully trained"):
        explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(2):
        for index in range(x.shape[0]):
            loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                             target=target, index=index)
            assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target, index=0)

    check_explanation(explanation, None, explainer.edge_mask_type)


@withCUDA
@pytest.mark.parametrize('mode', [
    ModelMode.binary_classification,
    ModelMode.multiclass_classification,
    ModelMode.regression,
])
def test_pg_explainer_graph(device, check_explanation, mode):
    x = paddle.randn(shape=[8, 3])
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], place=device)

    if mode == ModelMode.binary_classification:
        target = paddle.randint(2, (1, ), place=device)
    elif mode == ModelMode.multiclass_classification:
        target = paddle.randint(7, (1, ), place=device)
    elif mode == ModelMode.regression:
        target = paddle.randn(shape=[1, 1])

    model_config = ModelConfig(mode=mode, task_level='graph',
                               return_type='raw')

    model = GCN(model_config).to(device)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="not yet fully trained"):
        explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(2):
        loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                         target=target)
        assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target)

    check_explanation(explanation, None, explainer.edge_mask_type)


def test_pg_explainer_supports():
    # Test unsupported model task level:
    with pytest.raises(ValueError, match="not support the given explanation"):
        model_config = ModelConfig(
            mode='binary_classification',
            task_level='edge',
            return_type='raw',
        )
        Explainer(
            model=GCN(model_config),
            algorithm=PGExplainer(epochs=2),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=model_config,
        )

    # Test unsupported explanation type:
    with pytest.raises(ValueError, match="not support the given explanation"):
        model_config = ModelConfig(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        )
        Explainer(
            model=GCN(model_config),
            algorithm=PGExplainer(epochs=2),
            explanation_type='model',
            edge_mask_type='object',
            model_config=model_config,
        )

    # Test unsupported node mask:
    with pytest.raises(ValueError, match="not support the given explanation"):
        model_config = ModelConfig(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        )
        Explainer(
            model=GCN(model_config),
            algorithm=PGExplainer(epochs=2),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=model_config,
        )

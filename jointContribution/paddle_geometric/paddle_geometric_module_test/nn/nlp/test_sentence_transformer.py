import pytest

from paddle_geometric.nn.nlp import SentenceTransformer
from paddle_geometric.testing import onlyFullTest, withCUDA, withPackage


@withCUDA
@onlyFullTest
@withPackage('transformers')
@pytest.mark.parametrize('batch_size', [None, 1])
@pytest.mark.parametrize('pooling_strategy', ['mean', 'last', 'cls'])
def test_sentence_transformer(batch_size, pooling_strategy, device):
    model = SentenceTransformer(
        model_name='prajjwal1/bert-tiny',
        pooling_strategy=pooling_strategy,
    ).to(device)
    assert model.device == device
    assert str(model) == 'SentenceTransformer(model_name=prajjwal1/bert-tiny)'

    text = [
        "this is a basic english text",
        "PyG is the best open-source GNN library :)",
    ]

    out = model.encode(text, batch_size=batch_size)
    assert out.device == device
    assert out.shape== (2, 128)

    out = model.encode(text, batch_size=batch_size, output_place='cpu')
    assert out.is_cpu
    assert out.shape== (2, 128)

    out = model.encode([], batch_size=batch_size)
    assert out.device == device
    assert out.shape== (0, 128)

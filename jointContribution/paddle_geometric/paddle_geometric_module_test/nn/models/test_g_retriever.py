import paddle

from paddle_geometric.nn import GAT, GRetriever
from paddle_geometric.nn.nlp import LLM
from paddle_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_g_retriever() -> None:
    llm = LLM(
        model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1',
        num_params=1,
        dtype=paddle.float16,
    )

    gnn = GAT(
        in_channels=1024,
        out_channels=1024,
        hidden_channels=1024,
        num_layers=2,
        heads=4,
        norm='batch_norm',
    )

    model = GRetriever(
        llm=llm,
        gnn=gnn,
        mlp_out_channels=2048,
    )
    assert str(model) == ('GRetriever(\n'
                          '  llm=LLM(TinyLlama/TinyLlama-1.1B-Chat-v0.1),\n'
                          '  gnn=GAT(1024, 1024, num_layers=2),\n'
                          ')')

    x = paddle.randn(shape=[10, 1024])
    edge_index = paddle.to_tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])
    edge_attr = paddle.randn(shape=[edge_index.shape[1], 1024])
    batch = paddle.zeros(x.shape[0], dtype=paddle.int64)

    question = ["Is PyG the best open-source GNN library?"]
    label = ["yes!"]

    # Test train:
    loss = model(question, x, edge_index, batch, label, edge_attr)
    assert loss >= 0

    # Test inference:
    pred = model.inference(question, x, edge_index, batch, edge_attr)
    assert len(pred) == 1

import paddle

import paddle_geometric.typing
from paddle_geometric.nn import GraphSAGE
from paddle_geometric.profile.profiler import Profiler
from paddle_geometric.testing import withDevice


@withDevice
def test_profiler(capfd, get_dataset, device):
    x = paddle.randn(shape=[10, 16, place=device])
    edge_index = paddle.to_tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8],
    ], place=device)

    model = GraphSAGE(16, hidden_channels=32, num_layers=2).to(device)

    with Profiler(model, profile_memory=True, use_cuda=x.place.is_gpu_place()) as prof:
        model(x, edge_index)

    _, err = capfd.readouterr()

    _, heading_list, raw_results, layer_names, layer_stats = prof.get_trace()
    assert 'Self CPU total' in heading_list
    assert 'aten::relu' in raw_results
    assert '-act--aten::relu' in layer_names

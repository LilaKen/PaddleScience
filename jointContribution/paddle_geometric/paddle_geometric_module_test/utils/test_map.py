import pytest
import paddle

from paddle_geometric.profile import benchmark
from paddle_geometric.testing import withDevice, withPackage
from paddle_geometric.utils.map import map_index


@withDevice
@withPackage('pandas')
@pytest.mark.parametrize('max_index', [3, 100_000_000])
def test_map_index(device, max_index):
    src = paddle.to_tensor([2, 0, 1, 0, max_index], place=device)
    index = paddle.to_tensor([max_index, 2, 0, 1], place=device)

    out, mask = map_index(src, index, inclusive=True)
    assert out.device == device
    assert mask is None
    assert out.tolist() == [1, 2, 3, 2, 0]


@withDevice
@withPackage('pandas')
@pytest.mark.parametrize('max_index', [3, 100_000_000])
def test_map_index_na(device, max_index):
    src = paddle.to_tensor([2, 0, 1, 0, max_index], place=device)
    index = paddle.to_tensor([max_index, 2, 0], place=device)

    out, mask = map_index(src, index, inclusive=False)
    assert out.device == device
    assert mask.device == device
    assert out.tolist() == [1, 2, 2, 0]
    assert mask.tolist() == [True, True, False, True, True]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    src = paddle.randint(0, 100_000_000, (100_000, ), place=args.device)
    index = src.unique()

    def trivial_map(src, index, max_index, inclusive):
        if max_index is None:
            max_index = max(src.max(), index.max())

        if inclusive:
            assoc = src.new_empty(max_index + 1)
        else:
            assoc = src.new_full((max_index + 1, ), -1)
        assoc[index] = paddle.arange(index.numel(), place=index.device)
        out = assoc[src]

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    print('Inclusive:')
    benchmark(
        funcs=[trivial_map, map_index],
        func_names=['trivial', 'map_index'],
        args=(src, index, None, True),
        num_steps=100,
        num_warmups=50,
    )

    print('Exclusive:')
    benchmark(
        funcs=[trivial_map, map_index],
        func_names=['trivial', 'map_index'],
        args=(src, index[:50_000], None, False),
        num_steps=100,
        num_warmups=50,
    )

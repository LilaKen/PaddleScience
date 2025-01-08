import pytest


def test_ba_shapes(get_dataset):
    with pytest.warns(UserWarning, match="is deprecated"):
        dataset = get_dataset(name='BAShapes')

    assert str(dataset) == 'BAShapes()'
    assert len(dataset) == 1
    assert dataset.num_features == 10
    assert dataset.num_classes == 4

    data = dataset[0]
    assert len(data) == 5
    assert data.edge_index.shape[1] >= 1120
    assert data.x.shape == (700, 10)
    assert data.y.shape == (700, )
    assert data.expl_mask.sum() == 60
    assert data.edge_label.sum() == 960

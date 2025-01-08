def test_karate(get_dataset):
    dataset = get_dataset(name='KarateClub')
    assert str(dataset) == 'KarateClub()'
    assert len(dataset) == 1
    assert dataset.num_features == 34
    assert dataset.num_classes == 4

    assert len(dataset[0]) == 4
    assert dataset[0].edge_index.shape== (2, 156)
    assert dataset[0].x.shape== (34, 34)
    assert dataset[0].y.shape== (34, )
    assert dataset[0].train_mask.shape== (34, )
    assert dataset[0].train_mask.sum().item() == 4

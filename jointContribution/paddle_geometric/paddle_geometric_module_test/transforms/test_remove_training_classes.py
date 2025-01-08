import paddle

from paddle_geometric.data import Data
from paddle_geometric.transforms import RemoveTrainingClasses


def test_remove_training_classes():
    y = paddle.to_tensor([1, 0, 0, 2, 1, 3])
    train_mask = paddle.to_tensor([False, False, True, True, True, True])

    data = Data(y=y, train_mask=train_mask)

    transform = RemoveTrainingClasses(classes=[0, 1])
    assert str(transform) == 'RemoveTrainingClasses([0, 1])'

    data = transform(data)
    assert len(data) == 2
    assert paddle.equal(data.y, y)
    assert data.train_mask.tolist() == [False, False, False, True, False, True]

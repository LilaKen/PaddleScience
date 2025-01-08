import paddle

from paddle_geometric.datasets.icews import EventDataset
from paddle_geometric.loader import DataLoader
from paddle_geometric.nn import RENet
from paddle_geometric.testing import is_full_test


class MyTestEventDataset(EventDataset):
    def __init__(self, root, seq_len):
        super().__init__(root, pre_transform=RENet.pre_transform(seq_len))
        self.load(self.processed_paths[0])

    @property
    def num_nodes(self):
        return 16

    @property
    def num_rels(self):
        return 8

    @property
    def processed_file_names(self):
        return 'data.pdparams'

    def _download(self):
        pass

    def process_events(self):
        sub = paddle.randint(self.num_nodes, (64, ), dtype=paddle.int64)
        rel = paddle.randint(self.num_rels, (64, ), dtype=paddle.int64)
        obj = paddle.randint(self.num_nodes, (64, ), dtype=paddle.int64)
        t = paddle.arange(8, dtype=paddle.int64).view(-1, 1).repeat(1, 8).view(-1)
        return paddle.stack([sub, rel, obj, t], dim=1)

    def process(self):
        data_list = self._process_data_list()
        self.save(data_list, self.processed_paths[0])


def test_re_net(tmp_path):
    dataset = MyTestEventDataset(tmp_path, seq_len=4)
    loader = DataLoader(dataset, 2, follow_batch=['h_sub', 'h_obj'])

    model = RENet(dataset.num_nodes, dataset.num_rels, hidden_channels=16,
                  seq_len=4)

    if is_full_test():
        jit = paddle.jit.export(model)

    logits = paddle.randn(shape=[6, 6])
    y = paddle.to_tensor([0, 1, 2, 3, 4, 5])

    mrr, hits1, hits3, hits10 = model.test(logits, y)
    assert 0.15 < mrr <= 1
    assert hits1 <= hits3 and hits3 <= hits10 and hits10 == 1

    for data in loader:
        log_prob_obj, log_prob_sub = model(data)
        if is_full_test():
            log_prob_obj_jit, log_prob_sub_jit = jit(data)
            assert paddle.allclose(log_prob_obj_jit, log_prob_obj)
            assert paddle.allclose(log_prob_sub_jit, log_prob_sub)
        model.test(log_prob_obj, data.obj)
        model.test(log_prob_sub, data.sub)

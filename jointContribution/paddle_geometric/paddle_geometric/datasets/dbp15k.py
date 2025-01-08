import os
import os.path as osp
from typing import Callable, Dict, List, Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.data import HeteroData, InMemoryDataset
from paddle_geometric.data import (
    Data,
    InMemoryDataset,
    download_google_url,
    extract_zip,
)
from paddle_geometric.utils import sort_edge_index


class DBP15K(InMemoryDataset):
    r"""The DBP15K dataset from the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding"
    <https://arxiv.org/abs/1708.05045>`_ paper, where Chinese, Japanese and
    French versions of DBpedia were linked to its English version.
    Node features are given by pre-trained and aligned monolingual word
    embeddings from the `"Cross-lingual Knowledge Graph Alignment via Graph
    Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        pair (str): The pair of languages (:obj:`"en_zh"`, :obj:`"en_fr"`,
            :obj:`"en_ja"`, :obj:`"zh_en"`, :obj:`"fr_en"`, :obj:`"ja_en"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.HeteroData` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    file_id = '1ggYlYf2_kTyi7oF9g07oTNn3VDhjl7so'

    def __init__(
        self,
        root: str,
        pair: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert pair in ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']
        self.pair = pair
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['en_zh', 'en_fr', 'en_ja', 'zh_en', 'fr_en', 'ja_en']

    @property
    def processed_file_names(self) -> str:
        return f'{self.pair}.pdparams'

    def download(self) -> None:
        path = download_google_url(self.file_id, self.root, 'data.zip')
        extract_zip(path, self.root)
        os.unlink(path)

    def process(self) -> None:
        embs = {}
        with open(osp.join(self.raw_dir, 'sub.glove.300d')) as f:
            for i, line in enumerate(f):
                info = line.strip().split(' ')
                if len(info) > 300:
                    embs[info[0]] = paddle.to_tensor([float(x) for x in info[1:]])
                else:
                    embs['**UNK**'] = paddle.to_tensor([float(x) for x in info])

        g1_path = osp.join(self.raw_dir, self.pair, 'triples_1')
        x1_path = osp.join(self.raw_dir, self.pair, 'id_features_1')
        g2_path = osp.join(self.raw_dir, self.pair, 'triples_2')
        x2_path = osp.join(self.raw_dir, self.pair, 'id_features_2')

        x1, edge_index1, rel1, assoc1 = self.process_graph(
            g1_path, x1_path, embs)
        x2, edge_index2, rel2, assoc2 = self.process_graph(
            g2_path, x2_path, embs)

        train_path = osp.join(self.raw_dir, self.pair, 'train.examples.20')
        train_y = self.process_y(train_path, assoc1, assoc2)

        test_path = osp.join(self.raw_dir, self.pair, 'test.examples.1000')
        test_y = self.process_y(test_path, assoc1, assoc2)

        data = HeteroData()
        data['x1'].x = x1
        data['x2'].x = x2
        data['x1', 'edge_index1'].edge_index = edge_index1
        data['x2', 'edge_index2'].edge_index = edge_index2
        data['train_y'] = train_y
        data['test_y'] = test_y

        self.save(data, self.processed_paths[0])

    def process_graph(
        self,
        triple_path: str,
        feature_path: str,
        embeddings: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        g1 = read_txt_array(triple_path, sep='\t', dtype='int64')
        subj, rel, obj = g1.t()

        x_dict = {}
        with open(feature_path) as f:
            for line in f:
                info = line.strip().split('\t')
                info = info if len(info) == 2 else info + ['**UNK**']
                seq = info[1].lower().split()
                hs = [embeddings.get(w, embeddings['**UNK**']) for w in seq]
                x_dict[int(info[0])] = paddle.stack(hs, axis=0)

        idx = paddle.to_tensor(list(x_dict.keys()))
        assoc = paddle.full((int(idx.max()) + 1,), -1, dtype='int64')
        assoc[idx] = paddle.arange(idx.shape[0])

        subj, obj = assoc[subj], assoc[obj]
        edge_index = paddle.stack([subj, obj], axis=0)
        edge_index, rel = sort_edge_index(edge_index, rel)

        xs = list(x_dict.values())
        for i in x_dict.keys():
            xs[assoc[i]] = x_dict[i]
        x = paddle.nn.utils.pad_sequence(xs, batch_first=True, padding_value=0)

        return x, edge_index, rel, assoc

    def process_y(self, path: str, assoc1: Tensor, assoc2: Tensor) -> Tensor:
        row, col, mask = read_txt_array(path, sep='\t', dtype='int64').t()
        mask = mask.astype('bool')
        return paddle.stack([assoc1[row[mask]], assoc2[col[mask]]], axis=0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.pair})'

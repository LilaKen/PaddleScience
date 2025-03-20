from typing import Tuple

import paddle
import scipy.io as scio
import numpy as np

def plas_collate_fn(batch):
    shuffled_batch = []
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]
        num_timesteps = t.shape[0]
        permuted_indices = paddle.randperm(n=num_timesteps)
        t = t[permuted_indices]
        u = u[..., permuted_indices]
        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(axis=0)
            shuffled_t = t.unsqueeze(axis=0)
            shuffled_u = u.unsqueeze(axis=0)
            shuffled_a = a.unsqueeze(axis=0)
        else:
            shuffled_pos = paddle.concat(x=(shuffled_pos, pos.unsqueeze(
                axis=0)), axis=0)
            shuffled_t = paddle.concat(x=(shuffled_t, t.unsqueeze(axis=0)),
                                       axis=0)
            shuffled_u = paddle.concat(x=(shuffled_u, u.unsqueeze(axis=0)),
                                       axis=0)
            shuffled_a = paddle.concat(x=(shuffled_a, a.unsqueeze(axis=0)),
                                       axis=0)
    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)
    return shuffled_batch



def transpose_aux_func(dims, dim0, dim1):
    perm = list(range(dims))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


class UnitTransformer:

    def __init__(self, X):
        self.mean = X.mean(axis=(0, 1), keepdim=True)
        self.std = X.std(axis=(0, 1), keepdim=True) + 1e-08

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / self.std
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = tuple(X.shape)
                return (X * (self.std - 1e-08) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        elif inverse:
            orig_shape = tuple(X.shape)
            return (X * (self.std[:, component] - 1e-08) + self.mean[:,
                                                           component]).view(orig_shape)
        else:
            return (X - self.mean[:, component]) / self.std[:, component]


class PlasDataset(paddle.io.Dataset):
    def __init__(self,
                 input_keys: Tuple[str, ...],
                 label_keys: Tuple[str, ...],
                 weight_keys: Tuple[str, ...],
                 data_path,
                 ntrain=900,
                 ntest=80,
                 s1=101,
                 s2=31,
                 T=20,
                 Deformation=4,
                 r1=1,
                 r2=1,
                 mode='train'):
        # 数据路径与基本参数
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.data_path = data_path
        self.ntrain = ntrain
        self.ntest = ntest
        self.s1 = s1
        self.s2 = s2
        self.T = T
        self.Deformation = Deformation
        self.r1 = r1
        self.r2 = r2
        self.mode = mode

        # 计算s1和s2的调整尺寸
        self.s1 = int((s1 - 1) / r1 + 1)
        self.s2 = int((s2 - 1) / r2 + 1)

        # 加载数据
        data = scio.loadmat(self.data_path)
        self.input = paddle.to_tensor(data['input'], dtype='float32')
        self.output = paddle.to_tensor(data['output'], dtype='float32').transpose(
            perm=transpose_aux_func(paddle.to_tensor(data=data[
                'output'], dtype='float32').ndim, -2, -1))

        # 根据mode选择训练数据或测试数据
        if self.mode == 'train':
            self.x_data = self.input[:self.ntrain, ::self.r1][:, :self.s1].reshape(self.ntrain, self.s1, 1).tile(
                repeat_times=[1, 1, self.s2])
            self.y_data = self.output[:self.ntrain, ::self.r1, ::self.r2][:, :self.s1, :self.s2]
        elif self.mode == 'eval':
            self.x_data = self.input[-self.ntest:, ::self.r1][:, :self.s1].reshape(self.ntest, self.s1, 1).tile(
                repeat_times=[1, 1, self.s2])
            self.y_data = self.output[-self.ntest:, ::self.r1, ::self.r2][:, :self.s1, :self.s2]
        else:
            raise ValueError("Mode must be 'train' or 'eval'")

        # 重塑数据形状
        self.x_data = self.x_data.reshape(self.x_data.shape[0], -1, 1)
        self.y_data = self.y_data.reshape(self.y_data.shape[0], -1, self.Deformation, self.T)

        # 归一化
        self.x_normalizer = UnitTransformer(self.x_data)
        self.x_data = self.x_normalizer.encode(self.x_data)

        # 网格坐标
        x = np.linspace(0, 1, self.s1)
        y = np.linspace(0, 1, self.s2)
        x, y = np.meshgrid(x, y)
        self.pos = np.c_[x.flatten(), y.flatten()]
        self.pos = paddle.to_tensor(data=self.pos, dtype='float32').unsqueeze(axis=0)

        # 时间步长
        t = np.linspace(0, 1, self.T)
        t = paddle.to_tensor(data=t, dtype='float32').unsqueeze(axis=0)

        # 将坐标和时间步复制为训练或测试集
        if self.mode == 'train':
            self.pos_data = self.pos.tile(repeat_times=[self.ntrain, 1, 1])
            self.t_data = t.tile(repeat_times=[self.ntrain, 1])
        elif self.mode == 'eval':
            self.pos_data = self.pos.tile(repeat_times=[self.ntest, 1, 1])
            self.t_data = t.tile(repeat_times=[self.ntest, 1])

        print(f'{self.mode.capitalize()} dataloader is over.')

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # 返回训练或测试数据（根据mode）
        return (
            {self.input_keys[0]: (self.pos_data[idx], self.t_data[idx], self.x_data[idx],)},
            {self.label_keys[0]: self.y_data[idx]},
            {self.weight_keys[0]: paddle.to_tensor(1)},
        )

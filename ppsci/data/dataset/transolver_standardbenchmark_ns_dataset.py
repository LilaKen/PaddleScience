from typing import Tuple

import paddle
import numpy as np
import scipy.io as scio


class NavierStokesDataset(paddle.io.Dataset):
    def __init__(self,
                 input_keys: Tuple[str, ...],
                 label_keys: Tuple[str, ...],
                 weight_keys: Tuple[str, ...],
                 data_path,
                 ntrain,
                 ntest,
                 T_in,
                 T,
                 r=1,
                 h=None,
                 mode='train'):
        """
        Args:
            data_path (str): 数据路径.
            ntrain (int): 训练集样本数量.
            ntest (int): 测试集样本数量.
            T_in (int): 输入序列的时间步长.
            T (int): 输出序列的时间步长.
            r (int): 下采样的倍数.
            h (int): 目标图像的高度.
            mode (str): 选择数据集 ('train' 或 'eval')，默认为 'train'.
        """
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.data_path = data_path
        self.ntrain = ntrain
        self.ntest = ntest
        self.T_in = T_in
        self.T = T
        self.r = r
        self.h = h if h is not None else int((64 - 1) / r + 1)
        self.mode = mode  # 'train' or 'eval'

        # 加载数据
        data = scio.loadmat(self.data_path)
        u = data['u']

        # 根据mode选择训练集或测试集
        if self.mode == 'train':
            # 训练数据
            self.data_a = u[:self.ntrain, ::self.r, ::self.r, :self.T_in][:, :self.h, :self.h, :]
            self.data_u = u[:self.ntrain, ::self.r, ::self.r, self.T_in:self.T + self.T_in][:, :self.h, :self.h, :]
            # 重塑数据并转为tensor
            self.data_a = paddle.to_tensor(self.data_a.reshape(self.data_a.shape[0], -1, self.data_a.shape[-1]))
            self.data_u = paddle.to_tensor(self.data_u.reshape(self.data_u.shape[0], -1, self.data_u.shape[-1]))

            # 生成坐标
            x = np.linspace(0, 1, self.h)
            y = np.linspace(0, 1, self.h)
            x, y = np.meshgrid(x, y)
            self.pos = np.c_[x.flatten(), y.flatten()]
            self.pos = paddle.to_tensor(self.pos, dtype='float32').unsqueeze(axis=0)

            self.pos_data = self.pos_train = self.pos.tile(repeat_times=[self.ntrain, 1, 1])
        elif self.mode == 'eval':
            # 测试数据
            self.data_a = u[-self.ntest:, ::self.r, ::self.r, :self.T_in][:, :self.h, :self.h, :]
            self.data_u = u[-self.ntest:, ::self.r, ::self.r, self.T_in:self.T + self.T_in][:, :self.h, :self.h, :]
            # 重塑数据并转为tensor
            self.data_a = paddle.to_tensor(self.data_a.reshape(self.data_a.shape[0], -1, self.data_a.shape[-1]))
            self.data_u = paddle.to_tensor(self.data_u.reshape(self.data_u.shape[0], -1, self.data_u.shape[-1]))

            # 生成坐标
            x = np.linspace(0, 1, self.h)
            y = np.linspace(0, 1, self.h)
            x, y = np.meshgrid(x, y)
            self.pos = np.c_[x.flatten(), y.flatten()]
            self.pos = paddle.to_tensor(self.pos, dtype='float32').unsqueeze(axis=0)

            self.pos_data = self.pos_test = self.pos.tile(repeat_times=[self.ntest, 1, 1])
        else:
            raise ValueError("Mode should be either 'train' or 'eval'")

    def __len__(self):
        # 返回数据集的大小
        return self.ntrain if self.mode == 'train' else self.ntest

    def __getitem__(self, idx):
        # 根据索引返回数据
        if self.mode == 'train':
            return (
                {self.input_keys[0]: (self.pos_train[idx], self.data_a[idx])},
                {self.label_keys[0]: self.data_u[idx]},
                {self.weight_keys[0]: paddle.to_tensor(1)},
            )
        elif self.mode == 'eval':
            # idx = idx - self.ntrain  # Adjust for the test set
            return (
                {self.input_keys[0]: (self.pos_test[idx], self.data_a[idx])},
                {self.label_keys[0]: self.data_u[idx]},
                {self.weight_keys[0]: paddle.to_tensor(1)},
            )

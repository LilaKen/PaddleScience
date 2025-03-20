from typing import Tuple

import paddle
import numpy as np
import scipy.io as scio


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


class DarcyDataset(paddle.io.Dataset):
    def __init__(self,
                 input_keys: Tuple[str, ...],
                 label_keys: Tuple[str, ...],
                 weight_keys: Tuple[str, ...],
                 ntrain,
                 ntest,
                 train_path=None,
                 eval_path=None,
                 r=1,
                 mode='train',):
        """
        Args:
            data_path (str): Base data path.
            train_path (str): Path to the training data.
            test_path (str): Path to the testing data.
            ntrain (int): Number of training samples.
            ntest (int): Number of test samples.
            r (int): Downsampling factor.
            device (str): Device to use, either 'cpu' or 'gpu'.
            mode (str): Mode for data, either 'train' or 'eval'.
        """
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.r = r
        self.s = int((421 - 1) / self.r + 1)
        self.ntrain = ntrain
        self.ntest = ntest
        self.train_path = train_path
        self.eval_path = eval_path
        self.mode = mode

        # Load training and testing data
        train_data = scio.loadmat(self.train_path)
        test_data = scio.loadmat(self.eval_path)

        # Extract data from mat files
        self.x_train = train_data['coeff'][:self.ntrain, ::self.r, ::self.r][:, :self.s, :self.s]
        self.y_train = train_data['sol'][:self.ntrain, ::self.r, ::self.r][:, :self.s, :self.s]
        self.x_test = test_data['coeff'][:self.ntest, ::self.r, ::self.r][:, :self.s, :self.s]
        self.y_test = test_data['sol'][:self.ntest, ::self.r, ::self.r][:, :self.s, :self.s]

        self.x_train = self.x_train.reshape([self.ntrain, -1])
        self.x_test = self.x_test.reshape([self.ntest, -1])
        self.y_train = self.y_train.reshape([self.ntrain, -1])
        self.y_test = self.y_test.reshape([self.ntest, -1])


        # # Normalize data
        # self.x_normalizer = UnitTransformer(self.x_train)
        # self.y_normalizer = UnitTransformer(self.y_train)
        #
        # # Normalize the training and testing data
        # self.x_train = self.x_normalizer.encode(self.x_train)
        # self.y_train = self.y_normalizer.encode(self.y_train)
        # self.x_test = self.x_normalizer.encode(self.x_test)
        # self.y_test = self.y_normalizer.encode(self.y_test)


        # Grid coordinates (pos)
        x = np.linspace(0, 1, self.s)
        y = np.linspace(0, 1, self.s)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.flatten(), y.flatten()]
        self.pos = paddle.to_tensor(data=pos, dtype='float32').unsqueeze(axis=0)
        self.pos_train = self.pos.tile(repeat_times=[self.ntrain, 1, 1])
        self.pos_test = self.pos.tile(repeat_times=[self.ntest, 1, 1])

        self.x_train = paddle.to_tensor(self.x_train, dtype=paddle.float32)
        self.y_train = paddle.to_tensor(self.y_train, dtype=paddle.float32)
        self.pos_train = paddle.to_tensor(self.pos_train, dtype=paddle.float32)
        self.x_test = paddle.to_tensor(self.x_test, dtype=paddle.float32)
        self.y_test = paddle.to_tensor(self.y_test, dtype=paddle.float32)
        self.pos_test = paddle.to_tensor(self.pos_test, dtype=paddle.float32)

        if self.mode == 'train':
            self.x_data = self.x_train
            self.y_data = self.y_train
            self.pos_data = self.pos_train
        elif self.mode == 'eval':
            self.x_data = self.x_test
            self.y_data = self.y_test
            self.pos_data = self.pos_test

    def __len__(self):
        """Return the size of the dataset."""
        if self.mode == 'train':
            return len(self.x_train)
        elif self.mode == 'eval':
            return len(self.x_test)
        else:
            raise ValueError("Mode must be either 'train' or 'eval'")

    def __getitem__(self, idx):
        """Fetch the data and labels for the given index."""
        return (
            {self.input_keys[0]: (self.pos_data[idx], self.x_data[idx])},
            {self.label_keys[0]: self.y_data[idx]},
            {self.weight_keys[0]: paddle.to_tensor(1, dtype=paddle.float32)},
        )
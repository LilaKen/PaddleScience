from typing import Tuple

import paddle
import numpy as np


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


class ElasticityDataset(paddle.io.Dataset):
    def __init__(self,
                 input_keys: Tuple[str, ...],
                 label_keys: Tuple[str, ...],
                 weight_keys: Tuple[str, ...],
                 data_path=None,
                 ntrain=1000,
                 ntest=200,
                 mode='train'):
        """
        Initialize the dataset for elasticity data.

        Args:
            data_path (str): Path to the data directory.
            ntrain (int): Number of training samples.
            ntest (int): Number of testing samples.
            mode (str): 'train' or 'eval', specify which part of the dataset to load.
            device (str): The device to use ('cpu' or 'gpu').
        """
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.data_path = data_path
        self.ntrain = ntrain
        self.ntest = ntest
        self.mode = mode

        # Define data paths
        PATH_Sigma = self.data_path + '/Random_UnitCell_sigma_10.npy'
        PATH_XY = self.data_path + '/Random_UnitCell_XY_10.npy'

        # Load input data
        input_s = np.load(PATH_Sigma)
        input_s = paddle.to_tensor(data=input_s, dtype='float32').transpose(perm=[1, 0])

        input_xy = np.load(PATH_XY)
        input_xy = paddle.to_tensor(data=input_xy, dtype='float32').transpose(perm=[2, 0, 1])

        # Split the data into training and testing based on the mode
        if self.mode == 'train':
            self.s_data = input_s[:ntrain]
            self.xy_data = input_xy[:ntrain]
        elif self.mode == 'eval':
            self.s_data = input_s[-ntest:]
            self.xy_data = input_xy[-ntest:]
        else:
            raise ValueError("Mode must be 'train' or 'eval'")

        # Normalize data
        self.y_normalizer = UnitTransformer(self.s_data)
        self.s_data = self.y_normalizer.encode(self.s_data)

        print(f'{self.mode.capitalize()} dataloader is over.')

    def __len__(self):
        # Return the size of the dataset
        return len(self.s_data)

    def __getitem__(self, idx):
        # Return the (input, target) pair
        return (
            {self.input_keys[0]: (self.xy_data[idx], self.xy_data[idx])},
            {self.label_keys[0]: self.s_data[idx]},
            {self.weight_keys[0]: paddle.to_tensor(1, dtype=paddle.float32)},
        )


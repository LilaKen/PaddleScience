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


class PipeDataset(paddle.io.Dataset):
    def __init__(self,
                 input_keys: Tuple[str, ...],
                 label_keys: Tuple[str, ...],
                 weight_keys: Tuple[str, ...],
                 data_path,
                 ntrain=1000,
                 ntest=200,
                 N=1200,
                 r1=1,
                 r2=1,
                 mode='train'):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.DATA_PATH = data_path
        self.ntrain = ntrain
        self.ntest = ntest
        self.N = N
        self.r1 = r1
        self.r2 = r2
        self.mode = mode

        # s1, s2 are the downsampled sizes of the data
        self.s1 = int((129 - 1) / self.r1 + 1)
        self.s2 = int((129 - 1) / self.r2 + 1)

        # Load input and output data
        INPUT_X = self.DATA_PATH + '/Pipe_X.npy'
        INPUT_Y = self.DATA_PATH + '/Pipe_Y.npy'
        OUTPUT_Sigma = self.DATA_PATH + '/Pipe_Q.npy'

        inputX = np.load(INPUT_X)
        inputX = paddle.to_tensor(data=inputX, dtype='float32')

        inputY = np.load(INPUT_Y)
        inputY = paddle.to_tensor(data=inputY, dtype='float32')

        # Stack X and Y into one input tensor
        self.input = paddle.stack(x=[inputX, inputY], axis=-1)

        # Load output data
        output = np.load(OUTPUT_Sigma)[:, 0]
        self.output = paddle.to_tensor(data=output, dtype='float32')

        # Select the appropriate data based on mode
        if self.mode == 'train':
            self.x_data = self.input[:self.N][:self.ntrain, ::self.r1, ::self.r2][:, :self.s1, :self.s2]
            self.y_data = self.output[:self.N][:self.ntrain, ::self.r1, ::self.r2][:, :self.s1, :self.s2]
        elif self.mode == 'eval':
            self.x_data = self.input[:self.N][-self.ntest:, ::self.r1, ::self.r2][:, :self.s1, :self.s2]
            self.y_data = self.output[:self.N][-self.ntest:, ::self.r1, ::self.r2][:, :self.s1, :self.s2]
        else:
            raise ValueError("Mode must be 'train' or 'eval'")

        # Reshape data as needed
        self.x_data = self.x_data.reshape(self.x_data.shape[0], -1, 2)
        self.y_data = self.y_data.reshape(self.y_data.shape[0], -1)

        # Normalization
        self.x_normalizer = UnitTransformer(self.x_data)
        self.y_normalizer = UnitTransformer(self.y_data)

        self.x_data = self.x_normalizer.encode(self.x_data)
        self.y_data = self.y_normalizer.encode(self.y_data)

        print(f"{self.mode.capitalize()} dataloader is over.")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Return the data as a tuple (input, target)
        return (
            {self.input_keys[0]: (self.x_data[idx], self.x_data[idx])},
            {self.label_keys[0]: self.y_data[idx]},
            {self.weight_keys[0]: paddle.to_tensor(1, dtype=paddle.float32)},
        )

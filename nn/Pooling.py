import copy

import numpy as np

from .Parameter import Parameter
from .Module import Module


class Pooling(Module):
    def __init__(self, pool_size=2, mode="max"):
        self.inputs = []
        self.data = None
        self.pool_size = pool_size
        self.pool_stride = pool_size
        self.mode = mode
        self.max_pos = []
        self.gradient = None

    def forward(self, x):
        self.inputs.append(x)
        self.data = self.pooling()
        return self

    def backward(self, grad):
        self.gradient = copy.copy(self.inputs[0].data)
        in_row, in_col = np.shape(self.inputs[0].data)
        if self.mode == "max":
            for i in range(0, in_row):
                for j in range(0, in_col):
                    if self.max_pos.count((i, j)) == 0:
                        self.gradient[i][j] = 0
        elif self.mode == "mean":
            for i in range(0, in_row):
                for j in range(0, in_col):
                    self.gradient[i, j] = self.data[int(i/self.pool_size), int(j/self.pool_size)]/self.pool_size/self.pool_size
        if isinstance(self.inputs[0], Module):
            self.inputs[0].backward(self.gradient)

    def __call__(self, x):
        return self.forward(x)

    def pooling(self):
        """INPUTS:
                  inputMap - input array of the pooling layer
                  poolSize - X-size(equivalent to Y-size) of receptive field
                  poolStride - the stride size between successive pooling squares

           OUTPUTS:
                   outputMap - output array of the pooling layer

           Padding mode - 'edge'
        """
        # inputMap sizes
        in_row, in_col = np.shape(self.inputs[0].data)

        # outputMap sizes
        out_row, out_col = int(np.floor(in_row / self.pool_stride)), int(np.floor(in_col / self.pool_stride))
        row_remainder, col_remainder = np.mod(in_row, self.pool_stride), np.mod(in_col, self.pool_stride)
        if row_remainder != 0:
            out_row += 1
        if col_remainder != 0:
            out_col += 1
        outputMap = np.zeros((out_row, out_col))

        # padding
        temp_map = np.lib.pad(self.inputs[0].data, ((0, self.pool_size - row_remainder), (0, self.pool_size - col_remainder)), 'edge')

        # pooling
        if self.mode == "max":
            for r_idx in range(0, out_row):
                for c_idx in range(0, out_col):
                    startX = c_idx * self.pool_stride
                    startY = r_idx * self.pool_stride
                    poolField = temp_map[startY:startY + self.pool_size, startX:startX + self.pool_size]
                    poolOut = np.max(poolField)
                    temp = np.where(temp_map[startY:startY + self.pool_size, startX:startX + self.pool_size] == poolOut)
                    self.max_pos.append((startY+temp[0][0], startX+temp[1][0]))
                    outputMap[r_idx, c_idx] = poolOut
        elif self.mode == "mean":
            for r_idx in range(0, out_row):
                for c_idx in range(0, out_col):
                    startX = c_idx * self.pool_stride
                    startY = r_idx * self.pool_stride
                    poolField = temp_map[startY:startY + self.pool_size, startX:startX + self.pool_size]
                    poolOut = np.sum(poolField)
                    mean_value = poolOut / self.pool_size / self.pool_size
                    outputMap[r_idx, c_idx] = mean_value

        # retrun outputMap
        return outputMap

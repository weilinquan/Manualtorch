import copy

import numpy as np

from Test_Parameter import Test_Parameter
from Module import Module


class Pooling(Module):
    def __init__(self, row, col, poolsize, poolstride, mode="max"):
        self.parameter = Test_Parameter((row, col))
        self.inputs = []
        self.data = None
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.mode = mode
        self.maxpos = []

    def forward(self, x):
        self.inputs.append(x)
        if self.mode == "max":
            self.data = self.pooling()
        return self

    def backward(self, grad):
        temp = copy.copy(self.inputs)
        in_row, in_col = np.shape(self.inputs)
        for i in range(0, in_row):
            for j in range(0, in_col):
                if self.maxpos.count((i, j)) == 0:
                    temp[i][j] = 0
        self.parameter.gradient = temp
        if isinstance(self.inputs[0], Module):
            self.inputs[0].backward(self.parameter.gradient)

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
        in_row, in_col = np.shape(self.inputs)

        # outputMap sizes
        out_row, out_col = int(np.floor(in_row / self.poolstride)), int(np.floor(in_col / self.poolstride))
        row_remainder, col_remainder = np.mod(in_row, self.poolstride), np.mod(in_col, self.poolstride)
        if row_remainder != 0:
            out_row += 1
        if col_remainder != 0:
            out_col += 1
        outputMap = np.zeros((out_row, out_col))

        # padding
        temp_map = np.lib.pad(self.inputs, ((0, self.poolsize - row_remainder), (0, self.poolsize - col_remainder)), 'edge')

        # max pooling
        if self.mode == "max":
            for r_idx in range(0, out_row):
                for c_idx in range(0, out_col):
                    startX = c_idx * self.poolstride
                    startY = r_idx * self.poolstride
                    poolField = temp_map[startY:startY + self.poolsize, startX:startX + self.poolsize]
                    poolOut = np.max(poolField)
                    temp = np.where(self.inputs == poolOut)
                    self.maxpos.append((temp[0][0], temp[1][0]))
                    outputMap[r_idx, c_idx] = poolOut
        elif self.mode == "mean":
            for r_idx in range(0, out_row):
                for c_idx in range(0, out_col):
                    startX = c_idx * self.poolstride
                    startY = r_idx * self.poolstride
                    poolField = temp_map[startY:startY + self.poolsize, startX:startX + self.poolsize]
                    poolOut = np.sum(poolField)
                    outputMap[r_idx, c_idx] = poolOut / self.poolsize / self.poolsize

        # retrun outputMap
        return outputMap

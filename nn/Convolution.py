import numpy as np

from Test_Parameter import Test_Parameter
from Module import Module


def conv2d(data, kernel, padding=False):
    row, col = data.shape
    kernel_size = kernel.shape[0]
    pad_size = int(kernel_size / 2)

    if padding:
        padding_inputs = np.zeros([row + 2 * pad_size, col + 2 * pad_size], np.float32)
        padding_inputs[pad_size:-pad_size, pad_size:-pad_size] = data
        data = padding_inputs

    result = np.zeros(data.shape)
    row, col = data.shape
    for r in range(pad_size, row - pad_size):
        for c in range(pad_size, col - pad_size):
            cur_input = data[r - pad_size:r + pad_size + 1, c - pad_size:c + pad_size + 1]
            cur_output = cur_input * kernel
            conv_sum = np.sum(cur_output)
            result[r, c] = conv_sum
            print(result)
    final = result[pad_size:result.shape[0] - pad_size, pad_size:result.shape[1] - pad_size]
    return final


class Convolution(Module):
    def __init__(self, row, col, kernel):
        self.parameter = Test_Parameter((row, col))
        self.inputs = []
        self.data = None
        # self.kernel = np.ones([3, 3])
        self.kernel = kernel

    def forward(self, x):
        self.inputs.append(x)
        self.data = conv2d(self.kernel)
        return self

    def backward(self, grad):
        self.parameter.gradient = conv2d(grad, np.flip(self.kernel, axis=0), True)
        if isinstance(self.inputs[0], Module):
            self.inputs[0].backward(self.parameter.gradient)

    def __call__(self, x):
        return self.forward(x)

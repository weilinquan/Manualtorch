import numpy as np

from nn.Test_Parameter import Test_Parameter

from nn.Module import Module


def conv2d(data, kernel, padding=False):
    row, col = data.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size - 1

    if padding:
        padding_inputs = np.zeros([row + 2 * pad_size, col + 2 * pad_size], np.float32)
        padding_inputs[pad_size:-pad_size, pad_size:-pad_size] = data
        data = padding_inputs

    result = np.zeros(data.shape)
    row, col = data.shape
    # print(data)
    # for r in range(pad_size, row - pad_size):
    #     for c in range(pad_size, col - pad_size):
    #         cur_input = data[r - pad_size:r + pad_size + 1, c - pad_size:c + pad_size + 1]
    #         print(data)
    #         print(cur_input)
    #         cur_output = cur_input * kernel
    #         conv_sum = np.sum(cur_output)
    #         result[r, c] = conv_sum
    #         print(result)
    for r in range(pad_size, row):
        for c in range(pad_size, col):
            cur_input = data[r - pad_size:r + 1, c - pad_size:c + 1]
            # print("cur_input")
            # print(cur_input)
            cur_output = cur_input * kernel
            # print("cur_output")
            # print(cur_output)
            conv_sum = np.sum(cur_output)
            result[r, c] = conv_sum
    # final = result[pad_size:result.shape[0] - pad_size, pad_size:result.shape[1] - pad_size]
    final = result[pad_size:row, pad_size:col]
    return final


class Convolution(Module):
    def __init__(self, row, col, kernel, bias=True):
        self.parameter = Test_Parameter((row, col))
        self.parameter.data = kernel
        self.inputs = []
        self.data = None
        self.bias = bias
        self.bias_params = None
        # if bias:
        #     self.bias_params = Test_Parameter((row, col))

        # self.kernel = np.ones([3, 3])
        self.name = "convolution"

    def forward(self, x):
        self.inputs.append(x)
        self.data = conv2d(x.data, self.parameter.data)
        if self.bias:
            x, y = x.data.shape
            self.bias_params = Test_Parameter((x-self.parameter.data.shape[0]+1, y-self.parameter.data.shape[1]+1))
        self.data = self.data + self.bias_params
        # *np.ones(self.data.shape)
        return self

    def backward(self, grad):
        self.parameter.gradient = conv2d(self.inputs[0].data, grad)
        self.bias_params.gradient = np.sum(grad)
        if isinstance(self.inputs[0], Module):
            self.inputs[0].backward(conv2d(grad, np.fliplr(np.flipud(self.parameter.data)), True))
        del self.inputs[0]

    def __call__(self, x):
        return self.forward(x)

from .Module import Module
import numpy
from .Parameter import Parameter


class Linear(Module):
    def __init__(self, row, col, bias=True):
        self.parameter = Parameter((row, col))
        self.bias=bias
        if bias:
            self.bias_params = Parameter((1, col))
        self.inputs = []
        self.data = None
        self.name = 'linear'

    def forward(self, x):
        self.inputs.append(x)
        self.data = numpy.matmul(x.data, self.parameter.data)
        if self.bias:
            self.data = self.data+self.bias_params.data
        return self

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad):
        self.parameter.gradient = numpy.matmul(self.inputs[0].data.T, grad)
        if self.bias:
            g = grad.copy()
            g = numpy.sum(g, axis=0)
            #g = g.reshape((g.shape[0], 1))
            self.bias_params.gradient = g
        if(isinstance(self.inputs[0], Module)):
            current_grad = numpy.matmul(grad, self.parameter.data.T)
            self.inputs[0].backward(current_grad)
        del self.inputs[0]

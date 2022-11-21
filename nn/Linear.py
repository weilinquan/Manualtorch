from .Module import Module
import numpy
from .Parameter import Parameter


class Linear(Module):
    def __init__(self, row, col):
        self.parameter = Parameter((row, col))
        self.inputs = []
        self.data = None

    def forward(self, x):
        self.inputs.append(x)
        self.data = numpy.matmul(x.data, self.parameter.data)
        return self

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad):
        self.parameter.gradient = numpy.matmul(self.inputs[0].data.T, grad)
        if(isinstance(self.inputs[0], Module)):
            current_grad = numpy.matmul(grad, self.parameter.data.T)
            self.inputs[0].backward(current_grad)

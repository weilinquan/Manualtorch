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
        self.parameter.gradient = numpy.matmul(self.inputs[0].data.T, self.data)
        self.parameter.gradient = numpy.matmul(self.parameter.gradient, grad)
        if(isinstance(self.inputs[0], Module)):
            current_grad = numpy.matmul(self.data, self.parameter.data.T)
            current_grad = numpy.matmul(grad, current_grad)
            self.inputs[0].backward(current_grad)

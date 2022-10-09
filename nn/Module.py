from .Parameter import Parameter
import numpy


class Module:
    def __init__(self, row, col):
        self.parameter = Parameter((row, col))
        self.inputs = []
        self.gradient = numpy.zeros([row, col])
        self.data = None

    def forward(self, *args):
        for i in args:
            self.inputs.append(i)
        self.output = 0
        return self

    def backward(self, grad):
        self.gradient = grad
        for i in self.inputs:
            i.backward(self.gradient)

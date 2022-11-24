from symbol import parameters
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
            if(isinstance(i, Module)):
                i.backward(grad)
    
    def parameters(self, attrs_dict):
        parameters_dict = {'parameter':[], 'gradient':[]}
        for k, v in attrs_dict.items():
            if isinstance(v, Parameter):
                parameters_dict['parameter'].append(v)
                parameters_dict['gradient'].append(v.gradient)
            if isinstance(v, Module):
                parameters_dict['parameter']+=v.parameters(v.__dict__)['parameter']
                parameters_dict['gradient']+=v.parameters(v.__dict__)['gradient']
        return parameters_dict

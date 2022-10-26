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
    
    def parameters(self):
        parameters_dict = {'parameter':[], 'gradient':[]}
        if hasattr(self, "parameter"):
            parameters_dict['parameter'].append(self.parameter)
            parameters_dict['gradient'].append(self.gradient)
        for module in self.inputs:
            if isinstance(module, Module):
                parameters_dict['parameter']+=module.parameters['parameter']
                parameters_dict['gradient']+=module.parameters['gradient']
        return parameters_dict

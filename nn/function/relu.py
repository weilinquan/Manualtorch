from ..Module import Module
import numpy
class ReLu(Module):
    def __init__(self):
        self.inputs = []
        self.data = None
        self.name = 'relu'
    
    def forward(self, x):
        self.inputs.append(x)
        self.data = x.data.copy()
        for idx, val in numpy.ndenumerate(self.data):
            self.data[idx] = max(0, self.data[idx])
        return self
    
    def backward(self, grad):
        for idx, val in numpy.ndenumerate(self.inputs[0].data):
            if self.inputs[0].data[idx] < 0:
                grad[idx] = 0
        if isinstance(self.inputs[0], Module):
            self.inputs[0].backward(grad)
        del self.inputs[0]
    
    def __call__(self, x):
        return self.forward(x)

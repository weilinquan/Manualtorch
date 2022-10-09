import sys
sys.path.append("..")
from nn.Linear import Linear
from Tensor import Tensor
import numpy


x = Tensor(numpy.ones([224, 224]), 'float32')
linear = Linear(224, 1)
result = 0
for i in linear.parameter.data:
    result += i
print(result)
print(linear(x).data)
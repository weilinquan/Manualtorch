import sys
sys.path.append("..")
import nn
from nn.Linear import Linear
from Tensor import Tensor
from nn.Lossfunction import MSELoss
from nn.optimizer.optimizer import SGD
import numpy
from matplotlib import pyplot
class network(nn.Module.Module):
    def __init__(self, row, col):
        self.inputs = None
        self.dense = Linear(row, col)
        self.dense2 = Linear(col, row)
    
    def forward(self, x):
        out = self.dense(x)
        out = self.dense2(out)
        return out
net = network(1, 224)
x = Tensor(numpy.random.random((224, 1)), 'float32')
y = x.data
loss_func_numpy = MSELoss()
for i in range(3000):
    out = net.forward(x)
    loss_numpy = loss_func_numpy(out, y)
    loss_func_numpy.backward()
    paras = out.parameters()
    sgd_numpy = SGD(0.000001)
    sgd_numpy(paras)
    if i % 100 == 0:
        pyplot.scatter(x.data, out.data, c='red')
        pyplot.scatter(x.data, y, c='blue')
        pyplot.show()
    print(loss_numpy)
result = 0
for i in linear.parameter.data:
    result += i
print(result)
print(linear(x).data)
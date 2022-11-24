import sys
sys.path.append("..")
from nn.function.relu import ReLu
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
        self.relu = ReLu()
        self.dense2 = Linear(col, row)
        self.name = 'network'
    
    def forward(self, x):
        out = self.dense(x)
        #print(out.parameter.data)
        #print(out.data)
        out = self.relu(out)
        #print(out.data)
        out = self.dense2(out)
        #print(out.parameter.data)
        return out
net = network(1, 224)
x = Tensor(numpy.random.random((224, 1)), 'float32')
y = numpy.square(x.data)
loss_func_numpy = MSELoss()
for i in range(3000):
    out = net.forward(x)
    loss_numpy = loss_func_numpy(out, y)
    loss_func_numpy.backward()
    paras = net.parameters(net.__dict__)
    sgd_numpy = SGD(0.01)
    sgd_numpy(paras)
    if i % 1000 == 0:
        pyplot.scatter(x.data, out.data, c='red')
        pyplot.scatter(x.data, y, c='blue')
        pyplot.show()
    print(loss_numpy)
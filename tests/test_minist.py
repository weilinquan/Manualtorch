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

class model(nn.Module.Module):
    def __init__(self):
        self.conv1 = 
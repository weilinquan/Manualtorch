import numpy as np
from nn.Convolution import Convolution, conv2d

inputs = np.ones([7, 5], int)
kernel = np.ones([3, 3], int)
kernel = np.array([[1, 2, 1],
                   [1, 1, 1],
                   [1, 1, 1]])
test = Convolution(5, 5)
out = conv2d(inputs, np.flip(kernel, axis=0), True)
print(out)



#import numpy as np
'''
loss函数这块传给你的还是计算图节点和label值，计算图节点的话就通过.value属性访问到预测值，label就是numpy数组
'''
'''
class Loss(object):

    def MSELoss(self,x,y:Module):#x是真实值，y是预测值
        assert len(x) == len(y.len())
        x = np.array(x)
        y = np.array(y)
        loss = np.sum(np.square(x - y.value)) / len(x)
        return loss

    def L1Loss(self,x,y:list):
        assert len(x) == len(y)
        x = np.array(x)
        y = np.array(y)
        loss = np.sum(np.abs(x - y)) / len(x)
        return loss

    def CrossEntropy_loss(self,y_true,y_pred:list):
        assert len(y_true) == len(y_pred)
        loss = 0
        for y, fx in zip(y_true, y_pred):
            loss += -y * np.log(fx)
        return loss

    def softmax(self,x):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        s = x_exp / x_sum
        return s

'''



import numpy as np


class MSELoss:
    def __init__(self):
        self.x = None
        self.y = None

    def __call__(self, x, y):
        self.x = x
        self.y = y
        return np.sum(np.square(x.data - y)) / x.data.size

    def backward(self):
        dx = 2 * (self.x.data - self.y) / self.x.data.size
        self.x.backward(dx)



class L2loss:
    def __init__(self):
        self.x=None
        self.y=None
    def __call__(self, x, y):
        self.x=x
        self.y=y
        return np.sum(np.square(x-y))
    def backward(self):
        dx=2*self.x
        return dx



class Entropy:
    def __init__(self):
        self.nx = None
        self.ny = None
        self.dnx = None

    def loss(self, nx, ny):
        self.nx = nx
        self.ny = ny
        loss = np.sum(- ny * np.log(nx))
        return loss

    def backward(self):
        self.dnx = - self.ny / self.nx
        return self.dnx

"""
0.116960116566
0.1169601165663142
----------
[ 0.034682 -0.000561 -0.029935  0.034016  0.021168 -0.000575  0.03608   0.019185
  0.012494 -0.002536 -0.040756 -0.015934 -0.004686 -0.041798  0.02092   0.031164
 -0.01721  -0.051175  0.020822  0.003614 -0.026012  0.02444   0.008264  0.036326
 -0.007696 -0.020748 -0.013576]
[ 0.034682 -0.000561 -0.029935  0.034016  0.021168 -0.000575  0.03608   0.019185
  0.012494 -0.002536 -0.040756 -0.015934 -0.004686 -0.041798  0.02092   0.031164
 -0.01721  -0.051175  0.020822  0.003614 -0.026012  0.02444   0.008264  0.036326
 -0.007696 -0.020748 -0.013576]
----------
[-0.034682  0.000561  0.029935 -0.034016 -0.021168  0.000575 -0.03608  -0.019185
 -0.012494  0.002536  0.040756  0.015934  0.004686  0.041798 -0.02092  -0.031164
  0.01721   0.051175 -0.020822 -0.003614  0.026012 -0.02444  -0.008264 -0.036326
  0.007696  0.020748  0.013576]
[-0.034682  0.000561  0.029935 -0.034016 -0.021168  0.000575 -0.03608  -0.019185
 -0.012494  0.002536  0.040756  0.015934  0.004686  0.041798 -0.02092  -0.031164
  0.01721   0.051175 -0.020822 -0.003614  0.026012 -0.02444  -0.008264 -0.036326
  0.007696  0.020748  0.013576]
"""


"""entropy
np.random.seed(123)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

entropy = Entropy()

x = np.random.random([5, 10])
y = np.random.random([5, 10])
x_tensor = torch.tensor(x, requires_grad=True)
y_tensor = torch.tensor(y, requires_grad=True)

loss_numpy = entropy.loss(x, y)
grad_numpy = entropy.backward()

loss_tensor = (- y_tensor * torch.log(x_tensor)).sum()
loss_tensor.backward()
grad_tensor = x_tensor.grad

print("Python Loss :", loss_numpy)
print("PyTorch Loss :", loss_tensor.data.numpy())

print("\nPython dx :")
print(grad_numpy)
print("\nPyTorch dx :")
print(grad_tensor.data.numpy())
"""


"""L2loss
np.random.seed(123)
np.set_printoptions(precision=3, suppress=True, linewidth=120)
x_numpy = np.random.random(27)
y_numpy = np.random.random(27)
print(x_numpy)
print("\n")
l2loss=L2loss()
loss_numpy = l2loss(x_numpy, y_numpy)
grad_numpy = l2loss.backward()
print("Python Loss :", loss_numpy)
print("\nPython dx :")
print(grad_numpy)
"""
#import numpy as np
'''
loss函数这块传给你的还是计算图节点和label值，计算图节点的话就通过.value属性访问到预测值，label就是numpy数组
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
        self.x.backward(dx)



class Entropy:
    def __init__(self):
        self.x = None
        self.y = None
        self.dx = None

    def loss(self, x, y):
        self.x = x
        self.y = y
        loss = np.sum(- y * np.log(x))
        return loss

    def backward(self):
        self.dx = - self.y / self.x
        self.x.backward(self.dx)


"""
np.random.seed(123)
np.set_printoptions(precision=6, suppress=True, linewidth=80)

x_numpy = np.random.random(27)
y_numpy = np.random.random(27)
x_torch = torch.tensor(x_numpy, requires_grad=True)
y_torch = torch.tensor(y_numpy, requires_grad=True)

loss_func_numpy = MSELoss()
loss_func_torch = torch.nn.MSELoss().float()

loss_numpy = loss_func_numpy(x_numpy, y_numpy)
loss_torch = loss_func_torch(x_torch, y_torch)

loss_torch.backward()
dx_numpy, dy_numpy = loss_func_numpy.backward()

print(loss_numpy)
print(loss_torch.data.numpy())
print("----------")
print(dx_numpy)
print(x_torch.grad.numpy())
print("----------")
print(dy_numpy)
print(y_torch.grad.numpy())

"""
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
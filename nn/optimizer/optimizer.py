

import numpy as np



class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, params):
        for i in range(len(params['parameter'])):
            params['parameter'][i].data -= self.lr * params['gradient'][i]
        #params -= self.lr * grads


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def __call__(self, params):
        if self.v is None:
            self.v = np.zeros_like(params['parameter'])

        for i in range(len(params['parameter'])):
            self.v = self.momentum * self.v + params['gradient'][i]
            params['parameter'][i].data -= self.lr * self.v

            """
            if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.momentum * self.v + grads
        params -= self.lr * self.v
            """

"""
调用方法：
momentum_numpy = Momentum(0.00001, 0.9)
momentum_numpy(params)
"""

class Adam:
    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):#betas直衰减率，在Adam中一般取0.9与0.999
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps#极小值，防止分母为0
        self.m = None#一阶矩
        self.v = None#二阶矩
        self.n = 0

    def __call__(self, params):
        if self.m is None:
            self.m = np.zeros_like(params['parameter'])
        if self.v is None:
            self.v = np.zeros_like(params['parameter'])

        self.n += 1

        for i in range(len(params['parameter'])):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * params['gradient'][i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(params['gradient'][i])

            alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
            alpha = alpha / (1 - np.power(self.beta1, self.n))

            params['parameter'][i].data -= alpha * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
            """
            
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        params -= alpha * self.m / (np.sqrt(self.v) + self.eps)
"""

"""
调用方法：
adam_numpy = Adam(lr=0.0001, betas=(0.9, 0.99), eps=0.001)
adam_numpy(params)
"""




"""

def check_optim(optim_numpy, optim_torch, p, p_torch):
    """
"""
    check with y = p * x^2
    optim param p
    """
"""
    x_size = 5
    x = np.random.random(x_size)
    x_torch = torch.tensor(x, requires_grad=True)

    dxi_numpy_list = []
    for i in range(x_size):
        yi_numpy = p * x[i] ** 2
        dxi_numpy = 2 * p * x[i]
        dxi_numpy_list.append(dxi_numpy)

        da = x[i] ** 2
        optim_numpy(p, da)

    for i in range(x_size):
        yi_torch = p_torch * x_torch[i] ** 2
        optim_torch.zero_grad()
        yi_torch.backward()
        optim_torch.step()

    print(np.array(dxi_numpy_list))
    print(x_torch.grad.data.numpy())


np.random.seed(123)
np.set_printoptions(precision=12, suppress=True, linewidth=80)

print("--- 检查SGD ---")
a_numpy = np.array(1.2)
a_torch = torch.tensor(a_numpy, requires_grad=True)
sgd_numpy = SGD(0.1)
sgd_torch = torch.optim.SGD([a_torch], lr=0.1)
check_optim(sgd_numpy, sgd_torch, a_numpy, a_torch)

print("--- 检查Momentum ---")
a_numpy = np.array(1.2)
a_torch = torch.tensor(a_numpy, requires_grad=True)
momentum_numpy = Momentum(0.1, 0.9)
momentum_torch = torch.optim.SGD([a_torch], lr=0.1, momentum=0.9)
check_optim(momentum_numpy, momentum_torch, a_numpy, a_torch)



print("--- 检查Adam ---")
a_numpy = np.array(1.2)
a_torch = torch.tensor(a_numpy, requires_grad=True)
adam_numpy = Adam(lr=0.1, betas=(0.9, 0.99), eps=0.001)
adam_torch = torch.optim.Adam([a_torch], lr=0.1, betas=(0.9, 0.99), eps=0.001)
check_optim(adam_numpy, adam_torch, a_numpy, a_torch)
"""

"""
--- 检查SGD ---
[ 1.671526045435  0.658974920984  0.518721027022  1.254968104394  1.594004424417]
[ 1.671526045435  0.658974920984  0.518721027022  1.254968104394  1.594004424417]
--- 检查Momentum ---
[ 1.015455504299  2.318718975892  1.46525696166   0.886671018481  0.600349866658]
[ 1.015455504299  2.318718975892  1.46525696166   0.886671018481  0.600349866658]
--- 检查RMSProp ---
[ 0.823627238762  1.288627900967  0.503750904131  0.055347017535  0.367439505846]
[ 0.823627238762  1.288627900967  0.503750904131  0.055347017535  0.367439505846]
--- 检查Adam ---
[ 1.771188973757  0.402139865001  0.361961132239  1.035028538823  0.96262110244 ]
[ 1.771188973757  0.402139865001  0.361961132239  1.035028538823  0.96262110244 ]
"""





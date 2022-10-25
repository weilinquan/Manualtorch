import numpy as np
'''
loss函数这块传给你的还是计算图节点和label值，计算图节点的话就通过.value属性访问到预测值，label就是numpy数组
'''

class Loss(object):

    def MSELoss(self,x,y:Node):#x是真实值，y是预测值
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


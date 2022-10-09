import sys
sys.path.append("..")
import Tensor


a = Tensor.Tensor([[1, 2, 3]], 'float32')
b = Tensor.Tensor(a.data, 'float32')
b.data = b.data.reshape(3, 1)
print(a*b)

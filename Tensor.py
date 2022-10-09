import numpy


class Tensor:
    def __init__(self, data, type):
        self.data = numpy.array(data, dtype=type)

    def __add__(self, other):
        return self.data+other

    def __sub__(self, other):
        return self.data-other

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return numpy.matmul(self.data, other.data)
        return self.data*other

    def reshape(self, row, col):
        self.data = self.data.reshape(row, col)
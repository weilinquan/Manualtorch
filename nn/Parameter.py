import numpy


class Parameter:
    def __init__(self, shape):
        self.data = numpy.random.normal(0, 1, tuple(shape))
        self.gradient = numpy.zeros(list(shape))

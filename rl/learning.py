from __future__ import division


class LearningRateException(Exception):
    pass


class LearningRate(object):
    def update(self):
        raise NotImplementedError


class LearningRate_geom(LearningRate):
    def __init__(self):
        self.reset()

    def reset(self):
        self.a = 1.

    def update(self):
        self.a /= self.a + 1


class LearningRate_const(LearningRate):
    def __init__(self, a):
        if a < 0. or a > 1.:
            raise LearningRateException('Constant learning rate must be in the range [0, 1] (is {}).'.format(a))
        self.a = a

    def reset(self):
        raise LearningRateException('Constant learning rate does not afford being reset.')

    def update(self):
        pass

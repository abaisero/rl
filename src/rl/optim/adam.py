import numpy as np


class Adam:
    def __init__(self, alpha=.001, beta1=.9, beta2=.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m1 = 0
        self.m2 = 0
        self.b1t = 1
        self.b2t = 1

    def __call__(self, grads):
        self.b1t *= self.beta1
        self.b2t *= self.beta2
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * grads
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * np.square(grads)
        m1_ = self.m1 / (1 - self.b1t)
        m2_ = self.m2 / (1 - self.b2t)
        # import ipdb
        # ipdb.set_trace()
        grads_ = self.alpha * m1_ / (np.power(m2_, .5) + self.eps)
        # grads_ = self.alpha * m1_ / (np.sqrt(m2_) + self.eps)

        return grads_

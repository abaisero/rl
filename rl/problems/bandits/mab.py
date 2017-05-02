import numpy as np
import numpy.random as rnd
from scipy.stats import norm

from .. import Action, Model, RLProblem
from . import Bandit

from pytk.util import argmax


###############################################################################
## BANDITS
###############################################################################


class BinaryBandit(Bandit):
    def __init__(self, p):
        super(BinaryBandit, self).__init__()
        # if not 0 <= p <= 1:
        #     raise BanditException('BinaryBandit parameter is not in interval [0, 1]. (given {}).'.format(p))
        self.p = p

        self.maxr = 1
        self.Er = p

    def sample_r(self):
        return int(rnd.uniform() < self.p)


class GaussianBandit(Bandit):
    def __init__(self, m, s):
        super(GaussianBandit, self).__init__()
        self.m = m
        self.s = s
        self.rv = norm(m, s)

        self.maxr = max(abs(m + 3 * s), abs(m - 3 * s))
        self.Er = m

    def sample_r(self):
        return self.rv.rvs()


class ChoiceBandit(Bandit):
    def __init__(self, choices):
        super(ChoiceBandit, self).__init__()
        self.choices = choices
        self.nchoices = len(choices)

        self.maxr = max(map(abs, choices))
        self.Er = np.mean(choices)

    def sample_r(self):
        ci = rnd.choice(self.nchoices)
        return self.choices[ci]


###############################################################################
## Multi Armed Bandits
###############################################################################


class MABModel(Model):
    def __init__(self, bandits):
        for bidx, b in enumerate(bandits):
            b.bidx = bidx
        self.bandits = bandits

    def sample_r(self, b):
        return b.sample_r()


class MAB(RLProblem):
    def __init__(self, bandits):
        super(MAB, self).__init__(MABModel(bandits))
        self.actionlist = bandits

        # NOTE these are properties because we might have non-stationary bandits
        # self.maxr = max(b.maxr for b in bandits)
        # self.optimb = argmax(lambda b: b.Er, bandits, all_=True)

    @property
    def maxr(self):
        return max(b.maxr for b in self.model.bandits)

    @property
    def optimb(self):
        return argmax(lambda b: b.Er, self.model.bandits, all_=True)

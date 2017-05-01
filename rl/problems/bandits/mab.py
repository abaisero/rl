import numpy.random as rnd
from scipy.stats import norm

from .. import Action, Model, RLProblem
from . import Bandit, BanditAction


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

    def sample_r(self):
        return int(rnd.uniform() < self.p)


class GaussianBandit(Bandit):
    def __init__(self, m, s):
        super(GaussianBandit, self).__init__()
        self.m = m
        self.s = s
        self.rv = norm(m, s)

        self.maxr = max(abs(m + 3 * s), abs(m - 3 * s))

    def sample_r(self):
        return self.rv.rvs()


class ChoiceBandit(Bandit):
    def __init__(self, choices):
        super(ChoiceBandit, self).__init__()
        self.choices = choices
        self.nchoices = len(choices)

        self.maxr = max(map(abs, choices))

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
        # self.nbandits = len(bandits)

    def sample_r(self, b):
        # bidx = self.bandits.index(b)
        # return self.bandits[bidx].sample_r()
        return b.sample_r()


class MAB(RLProblem):
    def __init__(self, bandits):
        super(MAB, self).__init__(MABModel(bandits))
        # self.actionlist = map(BanditAction, xrange(self.model.nbandits))
        self.actionlist = bandits

        self.maxr = max(b.maxr for b in self.model.bandits)

    # def sample_r(self, a):
    #     return self.model.sample_r(a)

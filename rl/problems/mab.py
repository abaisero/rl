import numpy.random as rnd
from scipy.stats import norm

from . import Action, Model, RLProblem


###############################################################################
## BANDITS
###############################################################################

class BanditException(Exception):
    pass


class Bandit(object):
    def sample_r(self):
        raise NotImplementedError


class BinaryBandit(Bandit):
    def __init__(self, p):
        # if not 0 <= p <= 1:
        #     raise BanditException('BinaryBandit parameter is not in interval [0, 1]. (given {}).'.format(p))
        self.p = p

    def sample_r(self):
        return (rnd.uniform() < self.p)


class GaussianBandit(Bandit):
    def __init__(self, m, s):
        self.m = m
        self.s = s
        self.rv = norm(m, s)

    def sample_r(self):
        return self.rv.rvs()


class ChoiceBandit(Bandit):
    def __init__(self, choices):
        self.choices = choices
        self.nchoices = len(choices)
    
    def sample_r(self):
        ci = rnd.choice(self.nchoices)
        return self.choices[ci]


###############################################################################
## Multi Armed Bandits
###############################################################################


class MABAction(Action):
    def __init__(self, ai):
        self.ai = ai

    def __hash__(self):
        return hash(self.ai)

    def __eq__(self, other):
        return self.ai == other.ai

    def __str__(self):
        return 'A({})'.format(self.ai)


class MABModel(Model):
    def __init__(self, bandits):
        self.bandits = bandits
        self.nbandits = len(bandits)

    def sample_r(self, a):
        return self.bandits[a.ai].sample_r()


class MAB(RLProblem):
    def __init__(self, bandits):
        super(MAB, self).__init__(MABModel(bandits))
        self.actionlist = map(MABAction, xrange(self.model.nbandits))

    def sample_r(self, a):
        return self.model.sample_r(a)

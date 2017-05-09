from __future__ import division

from collections import defaultdict
import math

import numpy as np
import numpy.random as rnd

from rl.problems import taction, SAPair

from pytk.util import argmax


Qtype = object()
Atype = object()


class PolicyException(Exception):
    pass


class Policy(object):
    def __init__(self, ptype):
        self.ptype = ptype

    @classmethod
    def Q(cls, *args, **kwargs):
        return cls(Qtype, *args, **kwargs)

    @classmethod
    def A(cls, *args, **kwargs):
        return cls(Atype, *args, **kwargs)

    def dist_sa(self, s, actions):
        raise NotImplementedError

    def dist_a(self, actions):
        return self.dist_sa(None, actions)

    def dist(self, *args, **kwargs):
        if self.ptype is Qtype:
            return self.dist_sa(*args, **kwargs)
        elif self.ptype is Atype:
            return self.dist_a(*args, **kwargs)
        else:
            raise PolicyException('PolicyType ({}) unknown.'.format(self.ptype))

    def sample_sa(self, s, actions):
        dist = self.dist_sa(s, actions)
        p = np.array([dist[a] for a in actions])

        ai = rnd.choice(len(actions), p=p)
        return actions[ai]

    def sample_a(self, actions):
        return self.sample_sa(None, actions)

    def sample(self, *args, **kwargs):
        if self.ptype is Qtype:
            return self.sample_sa(*args, **kwargs)
        elif self.ptype is Atype:
            return self.sample_a(*args, **kwargs)
        else:
            raise PolicyException('PolicyType ({}) unknown.'.format(self.ptype))


class Policy_random(Policy):
    def dist_sa(self, s, actions):
        pr_uniform = 1 / len(actions)
        return dict.fromkeys(actions, pr_uniform)

    def sample_sa(self, s, actions):
        ai = rnd.choice(len(actions))
        return actions[ai]


class Policy_Qbased(Policy):
    def __init__(self, ptype, Q):
        super(Policy_Qbased, self).__init__(ptype)
        self.Q = Q


class Policy_egreedy(Policy_Qbased):
    def __init__(self, ptype, Q, e=0.):
        super(Policy_egreedy, self).__init__(ptype, Q)
        self.e = e

    def dist_sa(self, s, actions):
        pr_base = self.e / len(actions)
        dist = dict.fromkets(actions, pr_base)

        optim_actions = self.Q.optim_actions_sa(s, actions)
        pr_optim = pr_base + (1 - self.e) / len(optim_actions)
        update = dict.fromkeys(optim_actions, pr_optim)

        dist.update(update)
        return dist

    def sample_sa(self, s, actions):
        if self.e < rnd.random():
            return self.Q.optim_action_sa(s, actions)
        ai = rnd.choice(len(actions))
        return actions[ai]


class Policy_softmax(Policy_Qbased):
    def __init__(self, ptype, Q, tau=1.):
        super(Policy_softmax, self).__init__(ptype, Q)
        self.tau = tau

    def dist_sa(self, s, actions):
        prefs = np.array([self.Q.value_sa(s, a) for a in actions])
        prs = np.exp(prefs / self.tau)
        prs /= prs.sum()

        return dict(zip(actions, prs))


class Policy_UCB(Policy):
    def __init__(self, ptype, mu, sg, beta=1.):
        super(Policy_UCB, self).__init__(ptype)
        self.mu = mu
        self.sg = sg
        self.beta = beta

    def ucb_sa(self, s, a):
        return self.mu(s, a) + np.nan_to_num(self.beta * self.sg(s, a))

    def dist_sa(self, s, actions):
        dist = dict.fromkeys(actions, 0.)
        a = self.sample_sa(s, actions)
        dist[a] = 1.
        return dist

    def sample_sa(self, s, actions):
        return argmax(lambda a: self.ucb_sa(s, a), actions, rnd_=True)

from __future__ import division

import math

import numpy as np
import numpy.random as rnd 


class Policy(object):
    def __init__(self, actions):
        self.actions = actions

    # def iter_a(self, s):
    #     for a in self.actions(s):
    #         return a, self.pr_a(s, a)

    # def pr_a(self, s, a):
    #     raise NotImplementedError

    # def sample_a(self, s):
    #     actions = self.actions(s)
    #     pr = [self.pr_a(s, a) for a in actions]
    #     ai = rnd.choice(len(actions), p=pr)
    #     return actions[ai]

    def sample_a(self, s):
        raise NotImplementedError


class Policy_random(Policy):
    # def pr_a(self, s, a):
    #     return 1 / len(self.actions(s))

    def sample_a(self, s):
        actions = self.actions(s)
        i = rnd.choice(len(actions))
        return actions[i]


class Policy_valuebased(Policy):
    def __init__(self, actions, values):
        super(Policy_valuebased, self).__init__(actions)
        self.values = values


class Policy_egreedy(Policy_valuebased):
    def __init__(self, actions, values, e=0.):
        super(Policy_egreedy, self).__init__(actions, values)
        self.e = e

    # def pr_a(self, s, a):
    #     actions = len(self.actions(s))
    #     return self.e / nactions + (1 - self.e if self[s] == a else 0)

    def sample_a(self, s):
        if self.e < rnd.random():
            return self.values.optim_action(s)
        actions = self.actions(s)
        ai = rnd.choice(len(actions))
        return actions[i]


class Policy_UCB1(Policy_valuebased):
    def __init__(self, actions, values, beta=1.):
        super(Policy_UCB1, self).__init__(actions, values)
        self.beta = beta

    def sample_a(self, s):
        actions = self.actions(s)
        mus = np.array([self.values(s, a) for a in actions])
        ns = np.array([self.values.n(s, a) for a in actions])

        _2logntot = 2 * math.log(ns.sum())
        with np.errstate(divide='ignore'):
            sigmas = np.sqrt(_2logntot / ns)
        ucbs = np.nan_to_num(mus + self.beta * sigmas)
        max_ucb = max(ucbs)
        max_actions = [a for a, a_ucb in zip(actions, ucbs) if a_ucb == max_ucb]

        ai = rnd.choice(len(max_actions))
        return max_actions[ai]

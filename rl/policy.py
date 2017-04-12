from __future__ import division

import math

import numpy as np
import numpy.random as rnd 

from rl.problems import tstate, taction, SAPair


def UCB_confidence_Q(sa, Q):
    def confidence(sa):
        san = Q.nupdates(sa)
        sn = Q.nupdates_s(sa.s)

        try:
            _2logn = 2 * math.log(sn)
        except ValueError:
            _2logn = -np.inf
        try:
            return math.sqrt(_2logn / san)
        except ZeroDivisionError:
            return np.inf

    if sa is None:
        return confidence
    else:
        return confidence(sa)


# TODO maybe different versions?
def UCB_confidence(sa, sa_nupdates, s_nupdates):
    san = sa_nupdates(sa)
    sn = s_nupdates(sa.s)

    try:
        _2logn = 2 * math.log(sn)
    except ValueError:
        _2logn = -np.inf
    try:
        return math.sqrt(_2logn / san)
    except ZeroDivisionError:
        return np.inf


class Policy(object):
    def sample_a(self, actions, s=None):
        if s is tstate:
            return taction


class Policy_random(Policy):
    def sample_a(self, actions, s=None):
        super(Policy_random, self).sample_a(actions, s)
        i = rnd.choice(len(actions))
        return actions[i]


class Policy_Qbased(Policy):
    def __init__(self, Q):
        self.Q = Q

class Policy_egreedy(Policy_Qbased):
    def __init__(self, Q, e=0.):
        super(Policy_egreedy, self).__init__(Q)
        self.e = e

    def sample_a(self, actions, s=None):
        super(Policy_egreedy, self).sample_a(actions, s)
        if self.e < rnd.random():
            return self.Q.optim_action(actions, s)
        ai = rnd.choice(len(actions))
        return actions[ai]


# class Policy_UCB1(Policy_Qbased):
#     def __init__(self, actions, Q, beta=1.):
#         super(Policy_UCB1, self).__init__(actions, Q)
#         self.beta = beta

#     def sample_a(self, s=None):
#         # TODO this does not work..

#         actions = self.actions(s)
#         mus = np.array([self.Q(s, a) for a in actions])
#         ns = np.array([self.Q.n(s, a) for a in actions])

#         try:
#             _2logntot = 2 * math.log(ns.sum())
#         except ValueError:
#             _2logntot = -np.inf
#         with np.errstate(all='ignore'):
#             sigmas = np.sqrt(_2logntot / ns)
#         ucbs = mus + np.nan_to_num(self.beta * sigmas)
#         max_ucb = max(ucbs)
#         max_actions = [a for a, a_ucb in zip(actions, ucbs) if a_ucb == max_ucb]

#         ai = rnd.choice(len(max_actions))
#         return max_actions[ai]


class Policy_UCB(Policy):
    def __init__(self, m, s, beta=1.):
        self.m = m
        self.s = s
        self.beta = beta

    def sample_a(self, actions, s=None):
        super(Policy_UCB, self).sample_a(actions, s)
        sas = [SAPair(s=s, a=a) for a in actions]
        mus = np.array(map(self.m, sas))
        sigmas = np.array(map(self.s, sas))
        ucbs = mus + np.nan_to_num(self.beta * sigmas)
        max_ucb = max(ucbs)
        max_actions = [a for a, a_ucb in zip(actions, ucbs) if a_ucb == max_ucb]

        ai = rnd.choice(len(max_actions))
        return max_actions[ai]


# TODO some policies depend on a state, others don't...

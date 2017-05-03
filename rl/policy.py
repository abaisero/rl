from __future__ import division

from collections import defaultdict
import math

import numpy as np
import numpy.random as rnd

from rl.problems import taction, SAPair


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


def UCB_confidence_Q(sa, Q):
    return UCB_confidence(sa, Q.nupdates_sa, Q.nupdates_s)


class Policy(object):
    def pr_a(self, actions, s=None, a=None):
        raise NotImplementedError

    def sample_a(self, actions, s=None):
        pr_a = self.pr_a(actions, s)
        p = np.array([pr_a[a] for a in actions])

        ai = rnd.choice(len(actions), p=p)
        return actions[ai]
    # TODO I don't think I need this
    # def sample_a(self, actions, s=None):
    #     if s.terminal:
    #         return taction


class Policy_random(Policy):
    def pr_a(actions, s=None, a=None):
        pr_dict = defaultdict(lambda: 0,
            dict(zip(actions, itt.repeat(1/len(actions))))
        )
        return pr_dict if a is None else pr_dict[a]

    def sample_a(self, actions, s=None):
        # TODO why is this here
        # super(Policy_random, self).sample_a(actions, s)
        i = rnd.choice(len(actions))
        return actions[i]


class Policy_Qbased(Policy):
    def __init__(self, Q):
        self.Q = Q


class Policy_egreedy(Policy_Qbased):
    def __init__(self, Q, e=0.):
        super(Policy_egreedy, self).__init__(Q)
        self.e = e

    def pr_a(self, actions, s=None, a=None):
        pr_dict = defaultdict(lambda: 0,
            dict(zip(actions, itt.repeat(self.e/len(actions))))
        )
        pr_dict[self.Q.optim_action(actions, s)] += 1 - self.e
        return pr_dict if a is None else pr_dict[a]

    def sample_a(self, actions, s=None):
        # TODO why is this here?
        # super(Policy_egreedy, self).sample_a(actions, s)
        if self.e < rnd.random():
            return self.Q.optim_action(actions, s)
        ai = rnd.choice(len(actions))
        return actions[ai]


# TODO change softmax to incorporate policy comparison stuff
# class Policy_softmax(Policy_Qbased):
#     def __init__(self, Q, temp=1.):
#         super(Policy_softmax, self).__init__(Q)
#         self.temp = temp

#     def sample_a(self, actions, s=None):
#         Qs = np.array([self.Q(SAPair(s, a)) for a in actions])
#         pr_a = np.exp(Qs / self.temp)
#         pr_a /= pr_a.sum()

#         ai = rnd.choice(len(actions), p=pr_a)
#         return actions[ai]


class Preference(object):
    def value(self, a):
        raise NotImplementedError

    def values(self, actions):
        return np.array(map(self.value, actions))

    def update_value(self, a, r):
        raise NotImplementedError


class Preference_P(Preference):
    def __init__(self, alpha, beta, ref=0.):
        super(Preference_P, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ref = ref

        self.p = defaultdict(lambda: 0., {})

    def value(self, a):
        return self.p[a]

    def update_target(self, a, r):
        rdiff = r - self.ref
        self.p[a] += self.beta * rdiff
        self.ref += self.alpha * rdiff


class Preference_Q(Preference):
    def __init__(self, Q, tau=1.):
        super(Preference_Q, self).__init__()
        self.Q = Q
        self.tau = tau

    def value(self, a):
        return self.Q(SAPair(a=a)) / self.tau

    def update_target(self, a, r):
        self.Q.update_target(SAPair(a=a), r)


class Policy_softmax(Policy):
    def __init__(self, pref):
        super(Policy_softmax, self).__init__()
        self.pref = pref

    def pr_a(self, actions, a=None):
        pr_a = np.exp(self.pref.values(actions))
        pr_a /= pr_a.sum()

        pr_dict = defaultdict(lambda: 0, dict(zip(actions, pr_a)))
        return pr_dict if a is None else pr_dict[a]


class Policy_UCB(Policy):
    def __init__(self, m, s, beta=1.):
        self.m = m
        self.s = s
        self.beta = beta

    def sample_a(self, actions, s=None):
        # TODO why is this here
        # super(Policy_UCB, self).sample_a(actions, s)
        sas = [SAPair(s, a) for a in actions]
        mus = np.array(map(self.m, sas))
        sigmas = np.array(map(self.s, sas))
        ucbs = mus + np.nan_to_num(self.beta * sigmas)
        max_ucb = max(ucbs)
        max_actions = [a for a, a_ucb in zip(actions, ucbs) if a_ucb == max_ucb]

        ai = rnd.choice(len(max_actions))
        return max_actions[ai]

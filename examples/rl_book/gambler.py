from __future__ import division

from collections import defaultdict

import numpy as np
import math

from pytk.decorators import memoizemethod
from pytk.util import argmax

import matplotlib.pyplot as plt

from rl.problems import State, Action, SAPair, tstate, taction, Model, System

# from rl.env import environment, model, ModelException
from rl.values import StateValues_Tabular
from rl.algo.dp import policy_iteration, value_iteration
# from rl.policy import Policy_egreedy


class GamblerState(State):
    discrete=True

    def __init__(self, cash, goal):
        self.cash = cash
        self.terminal = not 0 < cash < goal

    def __hash__(self):
        return hash(self.cash)

    def __eq__(self, other):
        return self.cash == other.cash

    def __str__(self):
        return 'S({})'.format(self.cash)


class GamblerAction(Action):
    discrete = True

    def __init__(self, cash):
        self.cash = cash

    def __hash__(self):
        return hash(self.cash)

    def __eq__(self, other):
        return self.cash == other.cash

    def __str__(self):
        return 'A({})'.format(self.cash)


# class gambler_model(model):
#     @memoizemethod
#     def P(self, a, s0, s1):
#         """ Probability of getting to state s1 from s0 if action a is taken. """
#         if a not in self.env.actions(s0):
#             raise ModelException('Action not available')
#         # if s1 == s0 + a:
#         #     return coinp
#         # elif s1 == s0 - a:
#         #     return 1 - coinp
#         # return 0
#         return coinp if s1 == s0 + a else 1 - coinp if s1 == s0 - a else 0

#     @memoizemethod
#     def R(self, a, s0, s1):
#         if a not in self.env.actions(s0):
#             raise ModelException('Action not available')
#         return 1 if s1 == goal else 0


class GamblerModel(Model):
    def __init__(self, goal, coinp):
        self.goal = goal
        self.coinp = coinp

    def pr_s1(self, s0, a, s1=None):
        def makestate(cash):
            return GamblerState(cash) if 0 < cash < self.goal else tstate

        pr_dict = defaultdict(lambda: 0, {
            makestate(s0.cash + a.cash): self.coinp,
            makestate(s0.cash - a.cash): 1 - self.coinp,
        })
        return pr_dict if s1 is None else pr_dict.get(s1, 0)

        # def makestate(cash):
        #     return GamblerState(cash, self.goal) if 0 < cash < self.goal else tstate

        # pr_dict = defaultdict(lambda: 0, {
        #     GamblerState(s0.cash + a.cash, self.goal): self.coinp,
        #     GamblerState(s0.cash - a.cash, self.goal): 1 - self.coinp,
        # })
        # return pr_dict if s1 is None else pr_dict.get(s1, 0)

    def E_r(self, s0, a, s1):
        # TODO how do I distinguish one terminal state from another?
        return 1 if s1 is tstate else 0


class GamblerSystem(System):
    def __init__(self, goal, coinp):
        super(GamblerSystem, self).__init__(GamblerModel(goal, coinp))
        self.goal = goal
        self.coinp = coinp

        self.statelist = map(GamblerState, xrange(1, self.goal))
        self.statelist_wterm = self.statelist + [tstate]
        self.actionlist = map(GamblerAction, xrange(1, self.goal))

    @memoizemethod
    def actions(self, s):
        if s is tstate:
            return [taction]
        maxbet = min(s.cash, self.goal - s.cash)
        return self.actionlist[:maxbet]

# class gambler_environment(environment):
#     def __init__(self, goal, coinp):
#         self.goal = goal
#         self.coinp = coinp

#         self.__states_begin = []
#         self.__states_middle = range(1, goal)
#         self.__states_terminal = [0, goal]

#     @memoizemethod
#     def states(self, begin=True, middle=True, terminal=False):
#         return begin * self.__states_begin \
#                 + middle * self.__states_middle \
#                 + terminal * self.__states_terminal

#     @memoizemethod
#     def actions(self, s):
#         return range(1, min(s, self.goal - s) + 1)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3, linewidth=np.inf)

    # goal = 100
    # coinp = .4

    # env = gambler_environment(goal, coinp)
    # mod = gambler_model(env)
    # V = statevalues(env)
    # policy = egreedy(env, 0)

    # gamma = 1

    # # dpmethod = policy_iteration(env, mod, policy, V, gamma)
    # dpmethod = value_iteration(env, mod, policy, V, gamma)
    # dpmethod.iteration()

    # values = map(V.__getitem__, env.states())
    # actions = map(policy.__getitem__, env.states())

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(range(1, len(values)+1), values)
    # plt.subplot(212)
    # plt.plot(range(1, len(values)+1), actions)
    # plt.show()


    goal = 100
    coinp = .4

    sys = GamblerSystem(goal, coinp)
    V = StateValues_Tabular()

    gamma = 1

    # # dpmethod = policy_iteration(env, mod, policy, V, gamma)
    # dpmethod = value_iteration(env, mod, policy, V, gamma)
    # dpmethod.iteration()

    # policy_iteration(sys, V, gamma)
    value_iteration(sys, V, gamma)

    values = map(V, map(SAPair, sys.statelist))

    # TODO
    # there's a problem with terminal states..
    # if I only have 1 tstate, then how to distinguish rewards?
    # if I have multiple tstates,
    # if I only have states, with terminal thing, how to?

    print values

    # TODO fix this
    # values = map(V.__getitem__, env.states())
    # actions = map(policy.__getitem__, env.states())

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(range(1, len(values)+1), values)
    # plt.subplot(212)
    # plt.plot(range(1, len(values)+1), actions)
    # plt.show()

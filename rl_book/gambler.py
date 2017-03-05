from __future__ import division

import numpy as np
import math

from pytk.decorators import memoizemethod
from pytk.util import argmax

import matplotlib.pyplot as plt

from rl.env import environment, model, ModelException
from rl.values import statevalues
from rl.algo.dp import policy_iteration, value_iteration
from rl.policy import egreedy


class gambler_environment(environment):
    def __init__(self, goal, coinp):
        self.goal = goal
        self.coinp = coinp

        self.__states_begin = []
        self.__states_middle = range(1, goal)
        self.__states_terminal = [0, goal]

    @memoizemethod
    def states(self, begin=True, middle=True, terminal=False):
        return begin * self.__states_begin \
                + middle * self.__states_middle \
                + terminal * self.__states_terminal

    @memoizemethod
    def actions(self, s):
        return range(1, min(s, self.goal - s) + 1)


class gambler_model(model):
    @memoizemethod
    def P(self, a, s0, s1):
        """ Probability of getting to state s1 from s0 if action a is taken. """
        if a not in self.env.actions(s0):
            # print s0, a, self.env.actions(s0)
            raise ModelException('Action not available')
        if s1 == s0 + a:
            return coinp
        elif s1 == s0 - a:
            return 1 - coinp
        return 0

    @memoizemethod
    def R(self, a, s0, s1):
        if a not in self.env.actions(s0):
            raise ModelException('Action not available')
        if s1 == goal:
            return 1
        return 0


# def policy_evaluation(V, policy):
#     print 'policy_evaluation'
#     while True:
#         delta = 0
#         for s0 in states:
#             a = policy(s0).a
#             v = bellman(a, s0)
#             delta = max(delta, abs(v - V[s0]))
#             V[s0] = v

#         if delta < 1e-7:
#             break

# def policy_improvement(V, policy):
#     print 'policy_improvement'
#     policy_stable = True
#     for s0 in states:
#         a = bellman_optim(s0)
#         policy_stable = policy_stable and (a == policy(s0).a)
#         policy(s0).a = a
#     return policy_stable

# def policy_iteration(V, policy):
#     policy_stable = False
#     i = 0
#     print i
#     print policy
#     while not policy_stable:
#         print i
#         policy_evaluation(V, policy)
#         print V
#         policy_stable = policy_improvement(V, policy)
#         print policy
#         i += 1


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3, linewidth=np.inf)

    goal = 101
    coinp = .4
    gamma = 1

    env = gambler_environment(goal, coinp)
    mod = gambler_model(env)
    V = statevalues(env)
    policy = egreedy(env, 0)

    piter = policy_iteration(env, mod, policy, V, gamma)
    piter.iteration()

    # viter = value_iteration(env, mod, policy, V, gamma)
    # viter.iteration()

    values = map(V.__getitem__, env.states())
    actions = map(policy.__getitem__, env.states())

    plt.figure()
    plt.subplot(211)
    plt.plot(values)
    plt.subplot(212)
    plt.plot(actions)
    plt.show()

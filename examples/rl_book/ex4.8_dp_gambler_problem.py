from __future__ import division

from collections import defaultdict

import numpy as np
import math

from pytk.decorators import memoizemethod
from pytk.util import argmax

import matplotlib.pyplot as plt

from rl.problems import State, Action, SAPair, taction, Model, System
from rl.values import Values_Tabular
from rl.algo.dp import policy_iteration, value_iteration


class GamblerState(State):
    discrete=True

    def __init__(self, cash, goal):
        cash = np.clip(cash, 0, goal)
        self.setkey((cash,))

        self.cash = cash
        self.terminal = not 0 < cash < goal

    def __str__(self):
        return 'S({})'.format(self.cash)


class GamblerAction(Action):
    discrete = True

    def __init__(self, cash):
        self.setkey((cash,))
        self.cash = cash

    def __str__(self):
        return 'A({})'.format(self.cash)


class GamblerModel(Model):
    def __init__(self, goal, coinp):
        self.goal = goal
        self.coinp = coinp

    def pr_s1(self, s0, a, s1=None):
        pr_dict = defaultdict(lambda: 0, {
            GamblerState(s0.cash + a.cash, self.goal): self.coinp,
            GamblerState(s0.cash - a.cash, self.goal): 1 - self.coinp,
        })
        return pr_dict if s1 is None else pr_dict[s1]

    def E_r(self, s0, a, s1):
        return 1 if s1.terminal and s1.cash == self.goal else 0


class GamblerSystem(System):
    def __init__(self, goal, coinp):
        super(GamblerSystem, self).__init__(GamblerModel(goal, coinp))
        self.goal = goal
        self.coinp = coinp

        self.statelist = [GamblerState(cash, goal) for cash in xrange(0, self.goal + 1)]
        self.actionlist = [GamblerAction(cash) for cash in xrange(1, self.goal)]

    @memoizemethod
    def actions(self, s):
        if s.terminal:
            return [taction]
        maxbet = min(s.cash, self.goal - s.cash)
        return self.actionlist[:maxbet]


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3, linewidth=np.inf)

    goal = 100
    coinp = .4

    sys = GamblerSystem(goal, coinp)
    sys.model.gamma = 1

    V = Values_Tabular.V()

    # TODO implement policy_iteration
    # policy_iteration(sys, V, gamma)
    value_iteration.V(V, sys, sys.model)

    states = [s.cash for s in sys.states()]
    values = [V(s) for s in sys.states()]
    actions = [V.optim_action(s, sys.actions(s), sys.model).cash for s in sys.states()]

    plt.figure()
    plt.subplot(211)
    plt.plot(states, values)
    plt.subplot(212)
    plt.plot(states, actions)
    plt.show()

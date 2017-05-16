from __future__ import division

from collections import defaultdict

import numpy as np

from pytk.decorators import memoizemethod

import matplotlib.pyplot as plt

from rl.problems import State, Action, taction, Dynamics, Task, Model, System
from rl.values import Values_Tabular
from rl.algo.dp import ValueIteration


class GamblerState(State):
    discrete = True

    def __init__(self, cash, goal):
        super(GamblerState, self).__init__()
        cash = np.clip(cash, 0, goal)
        self.setkey((cash,))

        self.cash = cash
        self.terminal = not 0 < cash < goal

    def __str__(self):
        return 'S({})'.format(self.cash)


class GamblerAction(Action):
    discrete = True

    def __init__(self, cash):
        super(GamblerAction, self).__init__()
        self.setkey((cash,))
        self.cash = cash

    def __str__(self):
        return 'A({})'.format(self.cash)


class GamblerDynamics(Dynamics):
    def __init__(self, goal, coinp):
        super(GamblerDynamics, self).__init__()
        self.goal = goal
        self.coinp = coinp

    def dist_s1(self, s0, a):
        dist_s1 = defaultdict(int)
        dist_s1[GamblerState(s0.cash + a.cash, self.goal)] = self.coinp
        dist_s1[GamblerState(s0.cash - a.cash, self.goal)] = 1 - self.coinp
        return dist_s1


class GamblerTask(Task):
    def __init__(self, goal):
        super(GamblerTask, self).__init__()
        self.goal = goal

    def E_r(self, s0, a, s1):
        return 1 if s1.terminal and s1.cash == self.goal else 0


class GamblerSystem(System):
    def __init__(self, goal, coinp):
        dyna = GamblerDynamics(goal, coinp)
        task = GamblerTask(goal)
        super(GamblerSystem, self).__init__(Model(dyna, task))
        self.goal = goal

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
    sys.model.task.gamma = 1.

    V = Values_Tabular.V()

    # TODO implement policy_iteration
    # policy_iteration(sys, V, gamma)
    ValueIteration.V(V, sys, sys.model).run()

    states = [s.cash for s in sys.states()]
    values = [V(s) for s in sys.states()]
    actions = [V.optim_action(s, sys.actions(s), sys.model).cash for s in sys.states()]

    plt.figure()
    plt.subplot(211)
    plt.plot(states, values)
    plt.subplot(212)
    plt.plot(states, actions)
    plt.show()

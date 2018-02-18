from __future__ import division

from collections import defaultdict

import numpy as np

from pytk.decorators import memoizemethod

import matplotlib.pyplot as plt

from rl.problems import State, Action, taction, Model, System
from rl.values import Values_Tabular
# from rl.algo.dp import policy_iteration, value_iteration
from rl.algo.dp import ValueIteration

import pytk.factory as factory

import rl.problems.models.mdp as mdp


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3, linewidth=np.inf)

    goal = 100
    coinp = .4

    svalues = list(range(goal+1))
    sfactory = factory.FactoryChoice(svalues)

    avalues = list(range(1, goal+1))
    afactory = factory.FactoryChoice(avalues)


    def viability(s, a):
        return a.value <= s.value
    env = mdp.Environment(sfactory, afactory, viability)

    # TODO how to handle actions which don't really make sense in a certain state?
    # consider also general POMDP case where we don't know the state.... sooooo
    # just let it happen, leave everything as it is, and give bad reward... ok

    # TODO I guess s0 model doesn't matter
    s0model = mdp.State0Model(env)
    @s0model.pr
    def pr_s0(s0):
        pass

    s1model = mdp.State1Model(env)
    @s1model.dist
    def dist_s1(s0, a):
        dist = defaultdict(int)

        if not env.viable(s0, a):
            dist[s0] = 1
            return dist

        s1value = np.clip(s0.value+a.value, 0, goal)
        s1 = sfactory.item(value=s1value)
        dist[s1] = coinp

        s1value = np.clip(s0.value-a.value, 0, goal)
        s1 = sfactory.item(value=s1value)
        dist[s1] = 1 - coinp

        return dist


    rmodel = mdp.RewardModel()
    @rmodel.E
    def E(s0, a, s1):
        return 1 if s1.value == goal else 0

    model = mdp.Model(env, s0model, s1model, rmodel)
    model.gamma = 1.  # TODO remove this!!  not part of a model!


    # TODO terminal is returned by step function!!!


    # def terminal(s):
    #     return not 0 < s.value < goal


    # class GamblerState(State):
    #     discrete = True

    #     def __init__(self, cash, goal):
    #         super(GamblerState, self).__init__()
    #         cash = np.clip(cash, 0, goal)
    #         self.setkey((cash,))

    #         self.cash = cash
    #         self.terminal = not 0 < cash < goal

    #     def __str__(self):
    #         return 'S({})'.format(self.cash)



    # class GamblerAction(Action):
    #     discrete = True

    #     def __init__(self, cash):
    #         super(GamblerAction, self).__init__()
    #         self.setkey((cash,))
    #         self.cash = cash

    #     def __str__(self):
    #         return 'A({})'.format(self.cash)


    # class GamblerModel(Model):
    #     def __init__(self, goal, coinp):
    #         super(GamblerModel, self).__init__()
    #         self.goal = goal
    #         self.coinp = coinp

    #     def pr_s1(self, s0, a, s1=None):
    #         pr_dict = defaultdict(int)
    #         s1value = np.clip(0, s0.value + a.cash, goal)
    #         pr_dict[sfactory.item(value=s1value)] = self.coinp
    #         s1value = np.clip(0, s0.value - a.cash, goal)
    #         pr_dict[sfactory.item(value=s1value)] = 1 - self.coinp
    #         return pr_dict if s1 is None else pr_dict[s1]

    #     def E_r(self, s0, a, s1):
    #         return 1 if terminal(s1) and s1.value == self.goal else 0


    # class GamblerSystem(System):
    #     def __init__(self, goal, coinp):
    #         super(GamblerSystem, self).__init__(GamblerModel(goal, coinp))
    #         self.goal = goal
    #         self.coinp = coinp

    #         # self.statelist = [GamblerState(cash, goal) for cash in range(0, self.goal + 1)]
    #         self.statelist = list(sfactory.items)
    #         self.actionlist = [GamblerAction(cash) for cash in range(1, self.goal)]

    #     @memoizemethod
    #     def actions(self, s):
    #         if terminal(s):
    #             return [taction]
    #         maxbet = min(s.value, self.goal - s.value)
    #         return self.actionlist[:maxbet]


    # sys = GamblerSystem(goal, coinp)
    # sys.model.gamma = 1

    V = Values_Tabular.V()

    # todo implement policy_iteration
    # policy_iteration(sys, V, gamma)
    ValueIteration.V(V, env, model).run(tol=.5)  # good enough
    # ValueIteration.V(V, env, model).run(tol=1e-1)

    states = [s.value for s in env.states]
    values = [V(s) for s in env.states]
    # TODO handle actionlist stuff differently, in light of POMDP stuff..
    # I shouldn't have to specify which actions to consider...
    actions = [V.optim_action(s, env.actions, model).value for s in env.states]

    plt.figure()

    plt.subplot(211)
    plt.title('Values')
    plt.plot(states, values)
    plt.xlabel('state')
    plt.ylabel('value')

    plt.subplot(212)
    plt.title('Optimal Action')
    plt.plot(states, actions)
    plt.xlabel('state')
    plt.ylabel('action')

    plt.show()

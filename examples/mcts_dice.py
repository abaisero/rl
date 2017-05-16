import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from rl.problems import State, Action, Dynamics, Task, Model
from rl.problems.mdp import MDP
from rl.values import Values_TabularCounted, Values_LinearBayesian
from rl.policy import Policy_random, Policy_UCB
from rl.algo.mc import MCTS

from pytk.util import true_every


class DiceState(State):
    discrete = True

    def __init__(self, npips, nrolls, terminal=False):
        self.setkey((npips, nrolls))

        self.npips = npips
        self.nrolls = nrolls
        self.terminal = terminal

        degree = 2
        self.phi = np.empty(degree + 2)
        self.phi[0] = nrolls
        self.phi[1:] = np.vander([npips], degree + 1)

    def __str__(self):
        return 'S(npips={}, nrolls={})'.format(self.npips, self.nrolls)


class DiceAction(Action):
    discrete = True

    def __init__(self, hit):
        self.setkey((hit,))

        self.hit = hit
        self.stand = not hit

        self.phi = np.array([self.hit, self.stand], dtype=np.int64)

    def __str__(self):
        return 'A(\'{}\')'.format('hit' if self.hit else 'stand')


class DiceDynamics(Dynamics):
    def __init__(self, roll):
        super(DiceDynamics, self).__init__()
        self.roll = roll

    def sample_s0(self):
        return DiceState(self.roll(), 0)

    def sample_s1(self, s0, a):
        return (DiceState(s0.npips, s0.nrolls, True)
                if a.stand
                else DiceState(self.roll(), s0.nrolls + 1))


class DiceTask(Task):
    def __init__(self, nfaces, reroll_r):
        super(DiceTask, self).__init__()
        self.reroll_r = reroll_r

        self.maxr = max(nfaces, -reroll_r)

    def sample_r(self, s0, a, s1):
        return s0.npips if a.stand else self.reroll_r


class DiceMDP(MDP):
    """ Dice game MDP. """
    def __init__(self, nfaces, reroll_r):
        dyna = DiceDynamics(lambda: rnd.choice(nfaces)+1)
        task = DiceTask(nfaces, reroll_r)
        super(DiceMDP, self).__init__(Model(dyna, task))

        self.statelist_start = [DiceState(npips, 0) for npips in xrange(1, nfaces + 1)]
        self.actionlist = map(DiceAction, [True, False])


def run(mdp, sm):
    root_values = {}

    actions = []
    for s0 in mdp.statelist_start:
        print 'state: {}'.format(s0)
        a, values = sm.run(s0, 10000, 100, verbose=True)
        for a_ in mdp.actions(s0):
            root_values[s0, a_] = values[a_]

        print 'action: {}'.format(a)
        print '---'
        actions.append((s0, a))

    print 'cache'
    for s in mdp.statelist_start:
        for a in mdp.actions(s):
            print '{}: {}, {}'.format((s, a), Q(s, a), Q.n(s, a))

    print 'optimal actions'
    for s, a in actions:
        print '{}: {}'.format(s, a)

    print 'optimal actions'
    for s in mdp.statelist_start:
        for nrolls in xrange(5):
            s_ = DiceState(s.npips, nrolls)
            a = sm.policy_greedy.sample_a(s_)
            print '{} ; {} ; {} ; {}'.format(s_, a, Q(s_, a), Q.confidence(s_, a))

    plt.title(type(sm))
    for (s, a), values in root_values.iteritems():
        if a != 'stand':
            plt.plot(values, label=str((s, a)))
    plt.legend(loc=0)


if __name__ == '__main__':
    mdp = DiceMDP(nfaces=6, reroll_r=-1)

    print 'MCTS'
    print '===='

    # NOTE tabular AV
    # Q = Values_TabularCounted.Q()
    # policy_tree = Policy_UCB.Q(Q.value_sa, Q.confidence_sa, beta=mdp.maxr)

    # NOTE linear bayesian AV
    Q = Values_LinearBayesian.Q(l2=100., s2=.1)
    policy_tree = Policy_UCB.Q(Q.value, Q.confidence, beta=mdp.model.task.maxr)

    policy_dflt = Policy_random.Q()
    mcts = MCTS(mdp, mdp.model, policy_tree, policy_dflt, Q=Q)

    # NOTE algorithm
    # algo = MCTS(mdp, mdp.model, policy_tree, policy_dflt, Q=Q)
    # algo = SARSA(mdp, mdp.model, policy, Q)  # Equivalent to SARSA_l(0.)
    # algo = SARSA_l(mdp, mdp.model, policy, Q, .5)
    # algo = Qlearning(mdp, mdp.model, policy, Q)

    nepisodes = 100

    verbose = true_every(100)
    for i in xrange(nepisodes):
        s0 = mdp.model.dynamics.sample_s0()
        mcts.run(s0, 1000, verbose=verbose.true)


    print
    print 'cache'
    for s in mdp.statelist_start:
        for a in mdp.actions(s):
            print '{}, {}: {:.2f}, {:.2f}'.format(s, a, Q(s, a), Q.confidence(s, a))

    print
    print 'optimal actions'
    for s in mdp.statelist_start:
        actions = mdp.actions(s)
        print '{}: {}'.format(s, Q.optim_action(s, actions))

    print
    print 'optimal actions'
    for s in mdp.statelist_start:
        for nrolls in xrange(5):
            s_ = DiceState(s.npips, nrolls)
            actions = mdp.actions(s_)
            a = Q.optim_action(s_, actions)
            print '{} ; {} ; {:.2f} ; {:.2f}'.format(s_, a, Q(s, a), Q.confidence(s, a))


    # try:
    #     print 'Q.m:\n{}'.format(Q.m)
    # except AttributeError:
    #     pass

    # ax1 = plt.subplot(121)
    # run(mdp, mcts)
    # print '==='

    # print 'TDSearch'
    # print '===='
    # Q = ActionValues(mdp)
    # policy = Policy_UCB1(mdp.actions, Q, beta=mdp.maxr)
    # tds = TDSearch(mdp, mdp.model, policy, Q=Q)

    # ax2 = plt.subplot(122, sharey=ax1)
    # run(mdp, tds)
    # print '==='

    # plt.show()

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from rl.problems import State, Action, SAPair, Model
from rl.problems.mdp import MDP
from rl.values import ActionValues_TabularCounted, ActionValues_LinearBayesian
from rl.policy import Policy_random, Policy_UCB, UCB_confidence_Q
from rl.algo.search import MCTS, TDSearch

from pytk.util import true_every


class DiceState(State):
    discrete = True

    def __init__(self, npips, nrolls, terminal=False):
        self.npips = npips
        self.nrolls = nrolls
        self.terminal = terminal

        degree = 2
        self.phi = np.empty(degree + 2)
        self.phi[0] = nrolls
        self.phi[1:] = np.vander([npips], degree + 1)

    def __hash__(self):
        return hash((self.npips, self.nrolls))

    def __eq__(self, other):
        return self.npips == other.npips and self.nrolls == other.nrolls

    def __str__(self):
        return 'S(npips={}, nrolls={})'.format(self.npips, self.nrolls)


class DiceAction(Action):
    discrete = True

    def __init__(self, hit):
        self.hit = hit
        self.stand = not hit

        # self.phi = np.array([self.hit, self.stand], dtype=np.float64)
        self.phi = np.array([self.hit, self.stand], dtype=np.int64)

    def __hash__(self):
        return hash(self.hit)

    def __eq__(self, other):
        return self.hit == other.hit

    def __str__(self):
        return 'A(\'{}\')'.format('hit' if self.hit else 'stand')


class DiceModel(Model):
    def __init__(self, roll, reroll_r):
        super(DiceModel, self).__init__()
        self.roll = roll
        self.reroll_r = reroll_r

    def sample_s0(self):
        return DiceState(self.roll(), 0)

    def sample_s1(self, s0, a):
        if a.stand:
            return DiceState(s0.npips, s0.nrolls, True)
        return DiceState(self.roll(), s0.nrolls + 1)

    def sample_r(self, s0, a, s1):
        return s0.npips if a.stand else self.reroll_r


class DiceMDP(MDP):
    """ Dice game MDP. """
    def __init__(self, nfaces, reroll_r):
        # model = DiceModel(lambda: rnd.choice(nfaces)+1, reroll_r)
        # super(DiceMDP, self).__init__(model)
        super(DiceMDP, self).__init__(DiceModel(lambda: rnd.choice(nfaces)+1, reroll_r))

        self.nfaces = nfaces
        self.maxr = max(abs(reroll_r), abs(nfaces))

        self.statelist_start = [DiceState(npips, 0) for npips in xrange(1, nfaces + 1)]
        self.actionlist = map(DiceAction, [True, False])


def run(mdp, sm):
    root_values = {}

    actions = []
    for s0 in mdp.statelist_start:
        print 'state: {}'.format(s0)
        # a, values = sm.run(s0, 50000)
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
    Q = ActionValues_TabularCounted()
    def confidence(sa): return UCB_confidence_Q(sa, Q)
    policy_tree = Policy_UCB(Q.value, confidence, beta=mdp.maxr)

    # NOTE linear bayesian AV
    # Q = ActionValues_LinearBayesian(l=100., s2=1.)
    # Q = ActionValues_LinearBayesian(l2=100., s2=.1)
    # policy_tree = Policy_UCB(Q.value, Q.confidence, beta=mdp.maxr)

    policy_dflt = Policy_random()
    mcts = MCTS(mdp, mdp.model, policy_tree, policy_dflt, Q=Q)

    for i in range(100):
        s0 = mdp.model.sample_s0()
        mcts.run(s0, 1000, verbose=true_every(100))

    # for s0 in mdp.statelist_start:
    # for s0 in mdp.statelist_start[::-1]:
    #     mcts.run(s0, 100000, 1000, verbose=True)

    print
    print 'cache'
    for s in mdp.statelist_start:
        for a in mdp.actions(s):
            sa = SAPair(s, a)
            print '{}, {}: {:.2f}, {:.2f}'.format(s, a, Q(sa), confidence(sa))

    print
    print 'optimal_actions'
    for s in mdp.statelist_start:
        actions = mdp.actions(s)
        print '{}: {}'.format(s, Q.optim_action(actions, s))

    print
    print 'optimal actions'
    for s in mdp.statelist_start:
        for nrolls in xrange(5):
            s_ = DiceState(s.npips, nrolls)
            actions = mdp.actions(s_)
            a = Q.optim_action(actions, s_)
            sa = SAPair(s_, a)
            print '{} ; {} ; {:.2f} ; {:.2f}'.format(s_, a, Q(sa), confidence(sa))


    try:
        print 'Q.m:\n{}'.format(Q.m)
    except AttributeError:
        pass

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

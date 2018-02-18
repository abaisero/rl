import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from rl.problems import State, Action, SAPair, Model
# from rl.problems.mdp import MDP
from rl.values import Values_TabularCounted, Values_LinearBayesian
from rl.policy.policy import Policy_random, Policy_UCB
from rl.algo.mc import MCTS
from rl.algo.td import SARSA, SARSA_l, Qlearning, Qlearning_l

import rl.problems.models.mdp as mdp

from pytk.util import true_every
import pytk.factory as factory

from collections import defaultdict


max_rolls = 100
nfaces = 6
reroll_r = -1

def roll():
    return rnd.choice(nfaces)+1


npips_values = list(range(1, nfaces+1))
nrolls_values = list(range(max_rolls))
sfactory = factory.FactoryJoint(
        npips = factory.FactoryChoice(npips_values),
        nrolls = factory.FactoryChoice(nrolls_values),
)

avalues = ['hit', 'stand']
afactory = factory.FactoryChoice(avalues)

def viability(s, a):
    return a.value == 'stand' or s.nrolls.value < max_rolls

env = mdp.Environment(sfactory, afactory, viability)

s0model = mdp.State0Model(env)
@s0model.dist_
def dist_s0():
    dist = defaultdict(int)
    for npips in range(1, nfaces+1):
        s0value = dict(npips=npips, nrolls=0)
        dist[sfactory.item(value=s0value)] = 1 / nfaces
    return dist

@s0model.sample_
def sample_s0():
    s0value = dict(npips=roll(), nrolls=0)
    return sfactory.item(value=s0value)

s1model = mdp.State1Model(env)
@s1model.sample_
def sample_s1(s0, a):
    if a.value == 'hit':
        s1value = dict(npips=roll(), nrolls=s0.value.nrolls+1)
        return sfactory.item(value=s1value), False
    else:
        return s0, True

rmodel = mdp.RewardModel()
@rmodel.sample_
def sample_r(s0, a, s1):
    return s0.npips.value if a.value == 'stand' else reroll_r
rmodel.max_r = max(abs(nfaces), abs(reroll_r))


model = mdp.Model(env, s0model, s1model, rmodel)
model.gamma = 1.  # TODO remove this!!  not part of a model!


# class DiceState(State):
#     discrete = True

#     def __init__(self, npips, nrolls, terminal=False):
#         self.setkey((npips, nrolls))

#         self.npips = npips
#         self.nrolls = nrolls
#         self.terminal = terminal

#         degree = 2
#         self.phi = np.empty(degree + 2)
#         self.phi[0] = nrolls
#         self.phi[1:] = np.vander([npips], degree + 1)

#     def __str__(self):
#         return f'S(npips={self.npips}, nrolls={self.nrolls})'


# class DiceAction(Action):
#     discrete = True

#     def __init__(self, hit):
#         self.setkey((hit,))

#         self.hit = hit
#         self.stand = not hit

#         self.phi = np.array([self.hit, self.stand], dtype=np.int64)

#     def __str__(self):
#         return f'A({"hit" if self.hit else "stand"})'


# # TODO Model?
# class DiceModel(Model):
#     def __init__(self, roll, reroll_r):
#         super(DiceModel, self).__init__()
#         self.roll = roll
#         self.reroll_r = reroll_r

#     def sample_s0(self):
#         return DiceState(self.roll(), 0)

#     def sample_s1(self, s0, a):
#         return (DiceState(s0.npips, s0.nrolls, True)
#                 if a.stand
#                 else DiceState(self.roll(), s0.nrolls + 1))

#     def sample_r(self, s0, a, s1):
#         return s0.npips if a.stand else self.reroll_r


# class DiceMDP(MDP):
#     """ Dice game MDP. """
#     def __init__(self, nfaces, reroll_r):
#         model = DiceModel(lambda: rnd.choice(nfaces)+1, reroll_r)
#         super(DiceMDP, self).__init__(model)

#         self.nfaces = nfaces
#         self.maxr = max(abs(reroll_r), abs(nfaces))

#         self.statelist_start = [DiceState(npips, 0) for npips in range(1, nfaces + 1)]
#         self.actionlist = map(DiceAction, [True, False])


def run(mdp, sm):
    root_values = {}

    actions = []
    # for s0 in mdp.statelist_start:
    for s0 in model.s0.dist_s0():
        print(f'state: {s0}')
        a, values = sm.run(s0, 10000, 100, verbose=True)
        for a_ in mdp.actions(s0):
            root_values[s0, a_] = values[a_]

        print(f'action: {a}')
        print('---')
        actions.append((s0, a))

    print('cache')
    # for s in mdp.statelist_start:
    for s0 in model.s0.dist_s0():
        for a in mdp.actions(s):
            print(f'{(s, a)}: {Q(s, a)}, {Q.n(s, a)}')

    print('optimal actions')
    for s, a in actions:
        print(f'{s}: {a}')

    print('optimal actions')
    # for s in mdp.statelist_start:
    for s0 in model.s0.dist_s0():
        for nrolls in range(5):
            s_ = DiceState(s.npips, nrolls)
            a = sm.policy_greedy.sample_a(s_)
            print(f'{s_} ; {a} ; {Q(s_, a)} ; {Q.confidence(s_, a)}')

    plt.title(type(sm))
    for (s, a), values in root_values.iteritems():
        if a != 'stand':
            plt.plot(values, label=str((s, a)))
    plt.legend(loc=0)


if __name__ == '__main__':
    import numpy.random as rnd
    rnd.seed(0)
    # mdp = DiceMDP(nfaces=6, reroll_r=-1)

    print('MCTS')
    print('====')

    # NOTE tabular AV
    Q = Values_TabularCounted.Q()
    # policy = Policy_UCB.Q(Q.value_sa, Q.confidence_sa, beta=env.maxr)
    policy = Policy_UCB.Q(Q.value_sa, Q.confidence_sa, beta=model.r.max_r)

    # NOTE linear bayesian AV
    # Q = Values_LinearBayesian.Q(l2=100., s2=.1)
    # policy = Policy_UCB.Q(Q.value, Q.confidence, beta=env.maxr)

    # policy_dflt = Policy_random.Q()
    # mcts = MCTS(env, model, policy_tree, policy_dflt, Q=Q)

    # NOTE algorithm
    # algo = SARSA(env, model, policy, Q)  # Equivalent to SARSA_l(0.)
    # algo = SARSA_l(env, model, policy, Q, .5)
    algo = Qlearning(env, model, policy, Q)

    # TODO
    # algo = MCTS(env, model, policy_tree, policy_dflt, Q=Q)


    nepisodes = 10000

    verbose = true_every(100)
    for i in range(nepisodes):
        s0 = model.s0model.sample_s0()
        algo.run(s0, verbose=verbose.true)


    print()
    print('cache')
    # for s in mdp.statelist_start:
    for s in model.s0model.dist_s0():
        for a in env.actions:
        # for a in mdp.actions(s):
            print(f'{s}, {a}: {Q(s, a):.2f}, {Q.confidence(s, a):.2f}')

    print()
    print('optimal actions')
    # for s in mdp.statelist_start:
    for s in model.s0.dist_s0():
        # actions = mdp.actions(s)
        actions = env.actions
        print(f'{s}: {Q.optim_action(s, actions)}')

    # print
    # print 'optimal actions'
    # for s in mdp.statelist_start:
    #     for nrolls in range(3):
    #         s_ = DiceState(s.npips, nrolls)
    #         actions = mdp.actions(s_)
    #         a = Q.optim_action(s_, actions)
    #         print '{} ; {} ; {:.2f} ; {:.2f}'.format(s_, a, Q(s, a), Q.confidence(s, a))


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

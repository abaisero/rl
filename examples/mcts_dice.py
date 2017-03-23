from __future__ import division

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

import pytk.itt as tkitt

from rl.env import State, Environment, Model
from rl.values import ActionValues
from rl.policy import Policy_UCB1, Policy_random, Policy_egreedy


class DiceState(State):
    def __init__(self, npips, nrolls):
        self.npips = npips
        self.nrolls = nrolls

    def __hash__(self):
        return hash((self.npips, self.nrolls))

    def __eq__(self, other):
        return self.npips == other.npips and self.nrolls == other.nrolls

    def __str__(self):
        return 'DiceState({}, {})'.format(self.npips, self.nrolls)

    def __repr__(self):
        return str(self)


class DiceEnv(Environment):
    """ Dice game environment. """
    def __init__(self, nfaces, reroll_r=-1):
        super(DiceEnv, self).__init__()

        self.nfaces = nfaces
        self.scale_r = nfaces
        self.reroll_r = reroll_r

        self.states_start = [DiceState(npips, 0) for npips in xrange(1, self.nfaces + 1)]
        self.model = DiceModel(self)

    def actions(self, s):
        return ['hit', 'stand']

    def roll(self):
        return rnd.choice(self.nfaces) + 1


class DiceModel(Model):
    def sample_s1(self, s0, a):
        return self.env.terminal if a == 'stand' else DiceState(self.env.roll(), s0.nrolls + 1)

    def sample_r(self, s0, a, s1):
        return float(s0.npips if a == 'stand' else self.env.reroll_r)


class MCTS_node(object):
    def __init__(self, data, meta=None, parent=None):
        if meta is None:
            meta = {}

        self.data = data
        self.meta = meta
        self.parent = parent
        self.children = {}

    @property
    def nchildren(self):
        return len(self.children)

    def add_child(self, data, meta=None):
        child = MCTS_node(data, meta=None, parent=self)
        self.children[data] = child
        return child


class MCTS_tree(object):
    def __init__(self, env):
        self.env = env
        self.root = None

    def reroot(self, s=None):
        self.root = MCTS_node(s)
        # for a in self.env.actions(s):
        #     self.root.add_child(a)
        return self.root

class MCTS(object):
    def __init__(self, env, model, policy_tree, policy_dflt, Q, gamma=1.):
        # if cache is None:
        #     cache = MCTS_cache()

        self.env = env
        self.model = model
        self.policy_tree = policy_tree
        self.policy_dflt = policy_dflt
        self.policy_greedy = Policy_egreedy(env.actions, Q)

        self.tree = MCTS_tree(env)
        self.Q = Q
        self.gamma = gamma

    def policy_optimal(self, s):
        return self.policy_greedy.sample_a(s)

    def run(self, sroot, budget):
        root_values = {a: [self.Q(sroot, a)] for a in self.env.actions(sroot)}

        self.tree.reroot(sroot)
        for i in xrange(budget):
            snode = self.tree.root

            # SELECTION
            # print '---'
            while True:
                s0 = snode.data
                if s0 is self.env.terminal \
                        or len(snode.children) < len(self.env.actions(s0)):
                    break

                a = self.policy_tree.sample_a(s0)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                # print '{}, {}, {}, {}'.format(s0, a, r, s1)
                snode.meta['r'] = r

                anode = snode.children[a]
                try:
                    snode = anode.children[s1]
                except KeyError:
                    snode = anode.add_child(s1)

            # EXPANSION
            if s0 is not self.env.terminal:
                actions = set(self.env.actions(s0)) - set(snode.children.itervalues())
                ai = rnd.choice(len(actions))
                a = tkitt.nth(ai, actions)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                snode.meta['r'] = r
                anode = snode.add_child(a)
                snode = anode.add_child(s1)

                s0 = s1

            # SIMULATION
            g, gammat = 0, 1
            while s0 is not self.env.terminal:
                a = self.policy_dflt.sample_a(s0)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                g += gammat * r 
                gammat *= self.gamma
                # print '{}, {}, {}, {}'.format(s0, a, r, s1)

                s0 = s1

            # BACKPROPAGATION
            while anode is not None:
                snode = anode.parent
                g = snode.meta['r'] + self.gamma * g
                self.Q.update(g, snode.data, anode.data)
                anode = snode.parent

            for a in self.env.actions(sroot):
                root_values[a].append(self.Q(sroot, a))

        # Select the root action with highest score
        return self.policy_optimal(sroot), root_values


class TDSearch(object):
    def __init__(self, env, model, policy_tree, policy_dflt, Q, gamma=1.):
        # if cache is None:
        #     cache = MCTS_cache()

        self.env = env
        self.model = model
        self.policy_tree = policy_tree
        self.policy_dflt = policy_dflt
        self.policy_greedy = Policy_egreedy(env.actions, Q)

        self.tree = MCTS_tree(env)
        self.Q = Q
        self.gamma = gamma

    def policy_optimal(self, s):
        return self.policy_greedy.sample_a(s)

    def run(self, sroot, budget):
        # TODO TDSearch doesn't really need a tree..
        root_values = {a: [self.Q(sroot, a)] for a in self.env.actions(sroot)}

        self.tree.reroot(sroot)
        for i in xrange(budget):
            snode = self.tree.root

            # SELECTION
            # print '---'
            while True:
                s0 = snode.data
                if s0 is self.env.terminal \
                        or len(snode.children) < len(self.env.actions(s0)):
                    break

                a = self.policy_tree.sample_a(s0)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                # print '{}, {}, {}, {}'.format(s0, a, r, s1)
                self.Q.update(self.TD_target(r, s1), s0, a)

                anode = snode.children[a]
                try:
                    snode = anode.children[s1]
                except KeyError:
                    snode = anode.add_child(s1)

            # EXPANSION
            if s0 is not self.env.terminal:
                actions = set(self.env.actions(s0)) - set(snode.children.itervalues())
                ai = rnd.choice(len(actions))
                a = tkitt.nth(ai, actions)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                self.Q.update(self.TD_target(r, s1), s0, a)

                anode = snode.add_child(a)
                snode = anode.add_child(s1)

                s0 = s1

            # SIMULATION
            g, gammat = 0, 1
            while s0 is not self.env.terminal:
                a = self.policy_dflt.sample_a(s0)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                self.Q.update(self.TD_target(r, s1), s0, a)
                # print '{}, {}, {}, {}'.format(s0, a, r, s1)

                s0 = s1

            # BACKPROPAGATION
            # NOTE no backpropagation in TDSearch

            for a in self.env.actions(sroot):
                root_values[a].append(self.Q(sroot, a))

        # Select the root action with highest score
        return self.policy_optimal(sroot), root_values

    def TD_target(self, r, s1):
        return r + self.gamma * max(self.Q(s1, a) for a in self.env.actions(s1))


if __name__ == '__main__':
    env = DiceEnv(nfaces=6, reroll_r=-1)

    def run(smclass):
        Q = ActionValues(env)
        policy_tree = Policy_UCB1(env.actions, Q, beta=env.scale_r)
        policy_dflt = Policy_random(env.actions)
        sm = smclass(env, env.model, policy_tree, policy_dflt, Q=Q)

        root_values = {}

        actions = []
        for s0 in env.states_start:
            print 'state: {}'.format(s0)
            # a, values = sm.run(s0, 50000)
            a, values = sm.run(s0, 10000)
            for a_ in env.actions(s0):
                root_values[s0, a_] = values[a_]

            print 'action: {}'.format(a)
            print '---'
            actions.append((s0, a))

        print 'cache'
        for s in env.states_start:
            for a in env.actions(s):
                print '{}: {}, {}'.format((s, a), Q(s, a), Q.n(s, a))

        print 'optimal actions'
        for s, a in actions:
            print '{}: {}'.format(s, a)

        print 'optimal actions'
        for s in env.states_start:
            for nrolls in xrange(5):
                s_ = DiceState(s.npips, nrolls)
                a = sm.policy_greedy.sample_a(s_)
                print '{} ; {} ; {} ; {}'.format(s_, a, Q(s_, a), Q.n(s_, a))

        plt.title(str(smclass))
        for (s, a), values in root_values.iteritems():
            if a != 'stand':
                plt.plot(values, label=str((s, a)))
        plt.legend(loc=0)

    
    print 'MCTS'
    print '===='
    ax1 = plt.subplot(121)
    run(MCTS)
    print '==='


    print 'TDSearch'
    print '===='
    ax2 = plt.subplot(122, sharey=ax1)
    run(TDSearch)
    print '==='

    plt.show()

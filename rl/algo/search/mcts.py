import numpy.random as rnd

import pytk.itt as tkitt

from rl.problems import SAPair
from rl.util import Tree
from rl.policy import Policy_egreedy


class MCTS(object):
    def __init__(self, mdp, model, policy_tree, policy_dflt, Q, gamma=1):
        # if cache is None:
        #     cache = MCTS_cache()

        self.mdp = mdp
        self.model = model
        self.policy_tree = policy_tree
        self.policy_dflt = policy_dflt
        self.policy_greedy = Policy_egreedy(Q)

        self.tree = Tree()
        self.Q = Q
        self.gamma = gamma

    # def run(self, sroot, budget, every, verbose=False):
    def run(self, sroot, budget, verbose=False):
        root_values = {a: [self.Q(SAPair(sroot, a))] for a in self.mdp.actions(sroot)}

        self.tree.reroot(sroot)
        for i in xrange(budget):
            snode = self.tree.root

            # SELECTION
            # TODO there has to be a better way.. without me knowing whether verbose is a bool or an instance which evaluates to a bool
            verbose_ = bool(verbose)  # need to instance it in case I use true_every
            if verbose_:
                print '---'
            while True:
                s0 = snode.data
                actions = self.mdp.actions(s0)
                if s0.terminal \
                        or len(snode.children) < len(actions):
                    break

                a = self.policy_tree.sample_a(actions, s0)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                if verbose_:
                    print '{}, {}, {}, {}'.format(s0, a, r, s1)
                snode.meta['r'] = r

                anode = snode.children[a]
                try:
                    snode = anode.children[s1]
                except KeyError:
                    snode = anode.add_child(s1)

            # EXPANSION
            if not s0.terminal:
                actions = set(self.mdp.actions(s0)) - set(snode.children.itervalues())
                ai = rnd.choice(len(actions))
                a = tkitt.nth(ai, actions)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                if verbose_:
                    print '{}, {}, {}, {}'.format(s0, a, r, s1)

                snode.meta['r'] = r
                anode = snode.add_child(a)
                snode = anode.add_child(s1)

                s0 = s1

            # SIMULATION
            g, gammat = 0., 1.
            while not s0.terminal:
                actions = self.mdp.actions(s0)
                a = self.policy_dflt.sample_a(actions, s0)
                s1 = self.model.sample_s1(s0, a)
                r = self.model.sample_r(s0, a, s1)

                g += gammat * r
                gammat *= self.gamma
                if verbose_:
                    print '{}, {}, {}, {}'.format(s0, a, r, s1)

                s0 = s1

            # BACKPROPAGATION
            while anode is not None:
                snode = anode.parent
                g = snode.meta['r'] + self.gamma * g
                self.Q.update_target(SAPair(snode.data, anode.data), g)
                anode = snode.parent
            if verbose_:
                print 'Total return: {}'.format(g)

            for a in self.mdp.actions(sroot):
                root_values[a].append(self.Q(SAPair(sroot, a)))

        # Select the root action with highest score
        actions = self.mdp.actions(sroot)
        return self.policy_greedy.sample_a(actions, sroot), root_values

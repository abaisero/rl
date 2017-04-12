import numpy.random as rnd

import pytk.itt as tkitt

from rl.util import Tree
from rl.policy import Policy_egreedy


# TODO TDSearch doesn't really need a tree..
# TODO read about and implement RAVE


# class TDSearch(object):
#     def __init__(self, env, model, policy_tree, policy_dflt, Q, gamma=1.):
#         self.env = env
#         self.model = model
#         self.policy_tree = policy_tree
#         self.policy_dflt = policy_dflt
#         self.policy_greedy = Policy_egreedy(env.actions, Q)

#         self.tree = Tree()
#         self.Q = Q
#         self.gamma = gamma

#     def run(self, sroot, budget):
#         # TODO TDSearch doesn't really need a tree..
#         root_values = {a: [self.Q(sroot, a)] for a in self.env.actions(sroot)}

#         self.tree.reroot(sroot)
#         for i in xrange(budget):
#             snode = self.tree.root

#             # SELECTION
#             # print '---'
#             while True:
#                 s0 = snode.data
#                 if s0 is self.env.terminal \
#                         or len(snode.children) < len(self.env.actions(s0)):
#                     break

#                 a = self.policy_tree.sample_a(s0)
#                 s1 = self.model.sample_s1(s0, a)
#                 r = self.model.sample_r(s0, a, s1)

#                 # print '{}, {}, {}, {}'.format(s0, a, r, s1)
#                 self.Q.update(self.TD_target(r, s1), s0, a)

#                 anode = snode.children[a]
#                 try:
#                     snode = anode.children[s1]
#                 except KeyError:
#                     snode = anode.add_child(s1)

#             # EXPANSION
#             if s0 is not self.env.terminal:
#                 actions = set(self.env.actions(s0)) - set(snode.children.itervalues())
#                 ai = rnd.choice(len(actions))
#                 a = tkitt.nth(ai, actions)
#                 s1 = self.model.sample_s1(s0, a)
#                 r = self.model.sample_r(s0, a, s1)

#                 self.Q.update(self.TD_target(r, s1), s0, a)

#                 anode = snode.add_child(a)
#                 snode = anode.add_child(s1)

#                 s0 = s1

#             # SIMULATION
#             g, gammat = 0, 1
#             while s0 is not self.env.terminal:
#                 a = self.policy_dflt.sample_a(s0)
#                 s1 = self.model.sample_s1(s0, a)
#                 r = self.model.sample_r(s0, a, s1)

#                 self.Q.update(self.TD_target(r, s1), s0, a)
#                 # print '{}, {}, {}, {}'.format(s0, a, r, s1)

#                 s0 = s1

#             # BACKPROPAGATION
#             # NOTE no backpropagation in TDSearch

#             for a in self.env.actions(sroot):
#                 root_values[a].append(self.Q(sroot, a))

#         # Select the root action with highest score
#         return self.policy_greedy.sample_a(sroot), root_values

#     def TD_target(self, r, s1):
#         return r + self.gamma * max(self.Q(s1, a) for a in self.env.actions(s1))


class TDSearch(object):
    def __init__(self, env, model, policy, Q, gamma=1.):
        self.env = env
        self.model = model
        self.policy = policy

        self.Q = Q
        self.gamma = gamma

    def run(self, sroot, budget, every):
        # TODO TDSearch doesn't really need a tree..
        root_values = {a: [self.Q(sroot, a)] for a in self.env.actions(sroot)}

        for i in xrange(budget):
            s0 = sroot
            if i % every == 0:
                print '---'
            while s0 is not self.env.terminal:
                a = self.policy.sample_a(s0)
                r, s1 = self.model.sample_rs1(s0, a)
                if i % every == 0:
                    print '{}\t{}\t{:.2f}\t{:.2f}\t{}\t{}'.format(s0, a, self.Q(s0, a), self.Q.confidence(s0, a), r, s1)

                target = r + self.gamma * self.Q.optim_value(s1)
                # TODO abstract away the type of target? MC target, TD, TDl
                self.Q.update(target, s0, a)

                for a in self.env.actions(sroot):
                    root_values[a].append(self.Q(sroot, a))

                s0 = s1

        # Select the root action with highest score
        return self.Q.optim_action(sroot), root_values

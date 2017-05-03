import numpy.random as rnd

import pytk.itt as tkitt

from rl.problems import SAPair


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
#                 if s0 is self.env.tstate \
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
#             if s0 is not self.env.tstate:
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
#             while s0 is not self.env.tstate:
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
    def __init__(self, mdp, model, policy, Q, gamma=1.):
        self.mdp = mdp
        self.model = model
        self.policy = policy
        self.Q = Q
        self.gamma = gamma

    def run(self, sroot, budget, verbose=False):
        # TODO TDSearch doesn't really need a tree..
        # root_values = {a: [self.Q(sroot, a)] for a in self.env.actions(sroot)}

        for i in xrange(budget):
            s0 = sroot
            actions0 = self.mdp.actions(s0)
            a0 = self.policy.sample_a(actions0, s0)

            verbose_ = bool(verbose)
            if verbose_:
                print '---'
            while not s0.terminal:
                s0, a0 = self.runonce(s0, a0, verbose_)

        # Select the root action with highest score
        actions = self.mdp.actions(sroot)
        return self.Q.optim_action(actions, sroot)
        # return self.Q.optim_action(sroot), root_values

    def runonce(self, s0, a0=None, verbose=False):
        if a0 is None:
            a0 = self.policy.sample_a(actions0, s0)

        if not s0.terminal:
            r, s1 = self.model.sample_rs1(s0, a0)
            actions1 = self.mdp.actions(s1)
            a1 = self.policy.sample_a(actions1, s1)

            if verbose:
                print '{}, {}, {}, {}, {}'.format(s0, a0, r, s1, a1)

            self.Q.update(r=r, gamma=self.gamma, s0=s0, a0=a0, s1=s1, a1=a1, actions=self.mdp.actions(s1))

        return s1, a1

from __future__ import division

import numpy as np


class GreedyAgent(object):
    def __init__(self, mdp, model, Q):
        self.mdp = mdp
        self.model = model
        self.Q = Q

    def run(self, s0, verbose=False):
        history = []
        while not s0.terminal:
            actions = self.mdp.actions(s0)
            a = self.Q.optim_action(actions, s0)
            r, s1 = self.model.sample_rs1(s0, a)

            if verbose:
                print '{}, {}, {}, {}'.format(s0, a, r, s1)

            history += [(s0, a, r, s1)]
            s0 = s1

        return history

    def evaluate(self, s0, budget, verbose=False):
        g = np.empty(budget)
        for i in xrange(budget):
            # s0 = self.mdp.model.sample_s0()
            h = self.run(s0, verbose)
            g[i] = sum(r for _, _, r, _ in h)
        return g.mean(), g.var()

from .policy import Policy

import numpy.random as rnd


class eGreedy(Policy):
    def __init__(self, env, Q, e=0.):
        super().__init__(env)
        self.Q = Q
        self.e = e
        self.pr_base = e / self.env.nactions

    def dist(self, s):
        amax = self.Q.optim_actions(s)
        pr_amax = self.pr_base + (1 - self.e) / len(amax)

        for a in self.env.actions:
            yield a, pr_amax if a in amax else self.pr_base

    # TODO problem;  if another method uses pr, it...
    def pr(self, s, a):
        amax = self.amax(s, all_=True)
        pr_amax = self.pr_base + (1 - self.e) / len(amax)

        return pr_amax if a in amax else self.pr_base

    def sample(self, s):
        if self.e < rnd.rand():
            return self.amax(s, rnd_=True)
        return rnd.choice(list(self.env.actions))

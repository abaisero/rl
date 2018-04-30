from .policy import Policy

import numpy.random as rnd


class Random(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.pa = 1 / self.env.nactions

    def dist(self, s):
        for a in self.env.actions:
            yield a, self.pa

    def pr(self, s, a):
        return self.pa

    def sample(self, s):
        return rnd.choice(list(self.env.actions))

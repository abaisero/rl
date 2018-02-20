from .policy import Policy

import numpy.random as rnd


class Random(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.pa = 1 / self.env.nactions

    def reset(self):
        pass

    def restart(self):
        pass

    def dist(self):
        for a in self.env.actions:
            yield a, self.pa

    def pr(self, a):
        return self.pa

    def sample(self):
        return rnd.choice(list(self.env.actions))

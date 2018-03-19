from .policy import Policy

import rl.misc.models as models


class Blind(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.amodel = models.Softmax(env.aspace)

    def reset(self):
        self.amodel.reset()

    def restart(self):
        pass

    def dist(self):
        return self.amodel.dist()

    def pr(self, a):
        return self.amodel.pr(a)

    def sample(self):
        return self.amodel.sample()

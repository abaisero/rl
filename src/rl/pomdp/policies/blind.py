from .policy import Policy

import pytk.factory.model as fmodel


class Blind(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.amodel = fmodel.Softmax(env.afactory)

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

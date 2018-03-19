import rl.misc.dists as dists
from pytk.optim import argmax


class ActionDistribution(dists.SpaceDistribution):
    def __init__(self, env):
        super().__init__(env.aspace)
        self.env = env

    # TODO I suppose this could have a bug.... max(pr) might not be the same as max(Q)...
    def amax(self, **kwargs):
        dist = dict(self.dist())
        return argmax(
            lambda a:  dist.get(a, 0.),
            self.env.actions,
            **kwargs
        )

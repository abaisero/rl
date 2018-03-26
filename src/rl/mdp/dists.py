import rl.misc.dists as dists
from pytk.optim import argmax


class ActionDistribution(dists.SpaceDistribution):
    def __init__(self, env):
        super().__init__(env.aspace, cond=(env.sspace,))
        self.env = env

    # TODO I suppose this could have a bug.... max(pr) might not be the same as max(Q)...
    def amax(self, s, **kwargs):
        dist = dict(self.dist(s))
        return argmax(
            lambda a:  dist.get(a, 0.),
            self.env.actions,
            **kwargs
        )

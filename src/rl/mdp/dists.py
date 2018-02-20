import pytk.factory.random as frnd
from pytk.util import argmax


class ActionDistribution(frnd.FactoryDistribution):
    def __init__(self, env):
        super().__init__(env.afactory, cond=(env.sfactory,))
        self.env = env

    # TODO I suppose this could have a bug.... max(pr) might not be the same as max(Q)...
    def amax(self, s, **kwargs):
        ddist = dict(self.dist(s))
        return argmax(
            lambda a:  ddist.get(a, 0.),
            self.env.actions,
            **kwargs
        )

import pytk.factory.random as frnd


class ActionDistribution(frnd.FactoryDistribution):
    def __init__(self, env):
        super().__init__(env.afactory)
        self.env = env

    def amax(self, **kwargs):
        ddist = dict(self.dist())
        return argmax(
            lambda a:  ddist.get(a, 0.),
            self.env.actions,
            **kwargs
        )

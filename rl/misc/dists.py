import pytk.factory.random as frnd


class State0Distribution(frnd.FactoryDistribution):
    def __init__(self, env):
        super().__init__(env.sfactory)
        self.env = env


class State1Distribution(frnd.FactoryDistribution):
    def __init__(self, env):
        super().__init__(env.sfactory, cond=(env.sfactory, env.afactory))
        self.env = env


class ObsDistribution(frnd.FactoryDistribution):
    def __init__(self, env):
        super().__init__(env.ofactory, cond=(env.sfactory, env.afactory, env.sfactory))
        self.env = env


class RewardDistribution(frnd.RealDistribution):
    def __init__(self, env):
        super().__init__(cond=(env.sfactory, env.afactory, env.sfactory))
        self.env = env

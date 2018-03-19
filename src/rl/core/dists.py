import rl.misc.dists as dists


class State0Distribution(dists.SpaceDistribution):
    def __init__(self, env):
        super().__init__(env.sspace)
        self.env = env


class State1Distribution(dists.SpaceDistribution):
    def __init__(self, env):
        super().__init__(env.sspace, cond=(env.sspace, env.aspace))
        self.env = env


class ObsDistribution(dists.SpaceDistribution):
    def __init__(self, env):
        super().__init__(env.ospace, cond=(env.sspace, env.aspace, env.sspace))
        self.env = env


class RewardDistribution(dists.ScalarDistribution):
    def __init__(self, env):
        super().__init__(cond=(env.sspace, env.aspace, env.sspace))
        self.env = env

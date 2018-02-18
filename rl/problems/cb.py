from . import Model_


class Environment:
    def __init__(self, sfactory, afactory):
        self.sfactory = sfactory
        self.afactory = afactory

    @property
    def states(self):
        return self.sfactory.items

    @property
    def nstates(self):
        return self.sfactory.nitems

    @property
    def actions(self):
        return self.afactory.items

    @property
    def nactions(self):
        return self.afactory.nitems


class StateModel(Model_):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def dist(self):
        try:
            return self._dist()
        except TypeError:
            raise NotImplementedError

    def pr(self, s0):
        """ probability p(s0) """
        try:
            return self._pr(s0)
        except TypeError:
            pass

        return self.dist()[s0]

        # if self._dist is not None:
        #     return self._dist[s.i]

    def sample(self):
        """ samples initial state s0 ~ p(s0) """
        try:
            return self._sample()
        except TypeError:
            raise NotImplementedError

    def E(self, s, a):
        try:
            return self._E(s, a)
        except TypeError:
            raise NotImplementedError


class RewardModel(Model_):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def dist(self, a):
        try:
            return self._dist(a)
        except TypeError:
            raise NotImplementedError

    def pr(self, a, r):
        try:
            return self._pr(a, r)
        except TypeError:
            return self.dist(a)[r]

    def sample(self, a):
        try:
            return self._sample(a)
        except TypeError:
            raise NotImplementedError


class Model:
    def __init__(self, env, s0model, rmodel):
        self.env = env
        self.s0 = s0model
        self.r = rmodel

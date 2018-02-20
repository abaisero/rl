from .value import V, Counts, Eligibility
import rl.misc.fmodels as fmodels

import logging
logger = logging.getLogger(__name__)


class Tabular(V):
    def __init__(self, env):
        super().__init__(env)
        self.vmodel = fmodels.Tabular((env.sfactory,), 0.)
        # self.cmodel = fmodels.Tabular((env.sfactory,), 0)

    def counts(self):
        return TabularCounts(self.env)

    def eligibility(self, gamma, l):
        return TabularEligibility(self.env, gamma, l)

    # def ncounts(self, s):
    #     return self.cmodel.value((s,))

    # def update_value(self, s, value):
    #     self.vmodel[s] = value
    #     self.cmodel[s] += 1

    # should the value structure itslef have semantic methods like update_target?
    # this sounds... annoying to handle... just let this be an interface to the model, which also creates eligibility "clone" structures
    # def update_target(self, s, target, alpha):
    #     logger.debug(f'update_target({s}, {target}, {alpha})')
    #     value = self.value(s)
    #     delta = target - value
    #     self.update_value(s, value + alpha * delta)


# class Tabular_l(V):
#     def __init__(self, env):
#         super().__init__(env)
#         self.vmodel = fmodels.Tabular((env.sfactory,), 0.)

#     @staticmethod
#     def eligibility(gamma, l):
#         return TabularEligibility(gamma, l)

#     def update_value(self, s, value):
#         self.vmodel.update_value((s,), value)
#         self.cmodel.update_dvalue((s,), 1)

#     def update_target(self, s, target, alpha):
#         delta = target - self.value(s)
#         self.vmodel.update_dvalue(s, alpha * delta)


class TabularCounts(Counts):
    def __init__(self, env):
        super().__init__()
        self.cmodel = fmodels.Tabular((env.sfactory,), 0)


class TabularEligibility(Eligibility):
    def __init__(self, env, gamma, l):
        super().__init__(gamma, l)
        self.emodel = fmodels.Tabular((env.sfactory,), 0.)

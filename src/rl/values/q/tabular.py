from .value import Q, Counts, Eligibility
import rl.misc.fmodels as fmodels

import logging
logger = logging.getLogger(__name__)


class Tabular(Q):
    def __init__(self, env):
        super().__init__(env)
        self.qmodel = fmodels.Tabular((env.sfactory, env.afactory), 0.)
        # self.fvalues = fmodels.Tabular((env.sfactory, env.afactory), 0.)
        # self.fcounts = fmodels.Tabular((env.sfactory, env.afactory), 0)

    def counts(self):
        return TabularCounts(self.env)

    def eligibility(self, gamma, l):
        return TabularEligibility(self.env, gamma, l)

    def optim_value(self, s):
        """ optimized implementation for Tabular (?) """
        return self.qmodel[s, :].max()

    # def ncounts(self, s, a):
    #     return self.fcounts.value((s, a))

    # def update_value(self, s, a, value):
    #     self.fvalues.update_value((s, a), value)
    #     self.fcounts.update_dvalue((s, a), 1)

    # def update_target(self, s, a, target, alpha):
    #     logger.debug(f'update_target({s}, {a}, {target}, {alpha})')
    #     value = self.value(s, a)
    #     delta = target - value
    #     self.update_value(s, a, value + alpha * delta)

    def confidence(self, s, a):
        n_sa = self.ncounts(s, a)
        n_s = self.ncounts(s, None).sum()

        try:
            _2logn = 2 * math.log(n_s)
        except ValueError:
            _2logn = -np.inf

        try:
            _2logn_div_n = _2logn / n_sa
        except ZeroDivisionError:
            return np.inf

        return math.sqrt(_2logn_div_n)


# class Tabular_l(Q):
#     def __init__(self, env, alpha, gamma, l):
#         super().__init__(env)
#         self.fmodel = fmodels.Tabular_l((env.sfactory, env.afactory), alpha, gamma, l)


class TabularCounts(Counts):
    def __init__(self, env):
        super().__init__()
        self.cmodel = fmodels.Tabular((env.sfactory, env.afactory), 0)


class TabularEligibility(Eligibility):
    def __init__(self, env, gamma, l):
        super().__init__(gamma, l)
        self.emodel = fmodels.Tabular((env.sfactory, env.afactory), 0.)

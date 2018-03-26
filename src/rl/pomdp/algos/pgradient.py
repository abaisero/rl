from .algo import Algo

import numpy as np


class PolicyGradient(Algo):
    def __init__(self, pg, step_size):
        super().__init__()
        self.pg = pg
        if step_size is None:
            step_size = optim.StepSize(1)

    def __repr__(self):
        return repr(self.pg)

    def episode(self, env, policy, nsteps):
        dparams = self.pg.episode(env, policy, nsteps)

        lim = .1
        lim2 = lim ** 2
        gnorm2 = np.sum(_.sum() for _ in dparams ** 2)
        if gnorm2 > lim2:
            gnorm = np.sqrt(gnorm2)
            dparams *= lim / gnorm

        policy.params += dparams

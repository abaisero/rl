import logging
logger = logging.getLogger(__name__)

from .algo import Algo

import numpy as np


class PolicySearch(Algo):
    def __init__(self, ps):
        super().__init__()
        self.ps = ps

    def __repr__(self):
        return repr(self.pg)

    def episode(self, env, policy, nsteps):
        params = self.ps.episode(env, policy, nsteps)
        policy.params = params

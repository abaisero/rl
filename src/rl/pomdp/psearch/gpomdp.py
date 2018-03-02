import logging
logger = logging.getLogger(__name__)

from .p import P

import numpy as np


class GPOMDP(P):
    """ "Infinite-Horizon Policy-Gradient Estimation" - J. Baxter, P.Bartlett """

    logger = logging.getLogger(f'{__name__}.GPOMDP')

    def __init__(self, policy, beta):
        super().__init__(policy)
        self.beta = beta

    def restart(self):
        self.za0 = 0
        self.da0 = 0
        self.za = 0
        self.da = 0

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t = context.t
        o = self.policy.context.o
        r = feedback.r

        if t == 0:
            self.za0 = self.beta * self.za0 + self.policy.a0model.dlogprobs(o, a)
            self.da0 += (r * self.za0 - self.da0) / (t+1)

            self.za = self.beta * self.za
            self.da += (r * self.za - self.da) / (t+1)
        else:
            self.za0 = self.beta * self.za0
            self.da0 += (r * self.za0 - self.da0) / (t+1)

            self.za = self.beta * self.za + self.policy.amodel.dlogprobs(o, a)
            self.da += (r * self.za - self.da) / (t+1)

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        d = np.empty(2, dtype=object)
        d[:] = self.da0, self.da
        return d

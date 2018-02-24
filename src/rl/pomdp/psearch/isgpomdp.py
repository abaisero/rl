import logging
logger = logging.getLogger(__name__)

from .p import P

import numpy as np


class IsGPOMDP(P):
    """ "Scaling Internal-State Policy-Gradient Methods for POMDPs" - D. Aberdeen, J. Baxter """

    logger = logging.getLogger(f'{__name__}.IsGPOMDP')

    def __init__(self, policy, beta):
        super().__init__(policy)
        self.beta = beta

    def restart(self):
        self.za = 0
        self.da = 0
        self.zo = 0
        self.do = 0

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t = context.t
        o, r = feedback.o, feedback.r
        n = self.policy.context.n
        n1 = self.policy.feedback(feedback).n1

        self.za = self.beta * self.za + self.policy.amodel.dlogprobs(n, a)
        self.da += (r * self.za - self.da) / (t+1)

        self.zo = self.beta * self.zo + self.policy.omodel.dlogprobs(n, a, o, n1)
        self.do += (r * self.zo - self.do) / (t+1)

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        d = np.empty(2, dtype=object)
        d[:] = self.da, self.do

        return d

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
        self.z = 0
        self.d = 0

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t = context.t
        o, r = feedback.o, feedback.r
        n = self.policy.context.n
        n1 = self.policy.feedback(feedback).n1

        self.z = self.beta * self.z + self.policy.dlogprobs(n, a, o, n1)
        self.d += (r * self.z - self.d) / (t+1)

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        return self.d

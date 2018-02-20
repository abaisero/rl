import logging
logger = logging.getLogger(__name__)

from .agent import Agent

import numpy as np


class GPOMDP(Agent):
    logger = logging.getLogger(f'{__name__}.GPOMDP')

    def __init__(self, name, env, policy, beta):
        super().__init__(name, env, policy)
        self.beta = beta

    def reset(self):
        self.policy.reset()

    def restart(self):
        self.policy.restart()

        self.z = 0
        self.d = 0

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t = context.t
        o = self.policy.context.o
        r = feedback.r

        self.z = self.beta * self.z + self.policy.amodel.dlogprobs(o, a)
        self.d += (r * self.z - self.d) / (t+1)

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        alpha = 1
        self.policy.amodel.params += alpha * self.d

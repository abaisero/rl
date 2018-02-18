import logging
logger = logging.getLogger(__name__)

from .agent import Agent

import numpy as np


class Blind(Agent):
    logger = logging.getLogger(f'{__name__}.Blind')

    # TODO implement step-variant!
    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        G = 0.

        for context, a, feedback in reversed(episode):
            t = context.t
            r = feedback.r

            G = r + self.env.gamma * G

            # TODO let some other method take care of how to do the gradient descent (normalized or not, etc...)
            alpha = .001
            dparams = self.policy.amodel.dlogprobs(a)
            # dparams /= np.sqrt((dparams * dparams).sum())
            self.policy.amodel.params += alpha * G * dparams
            # self.policy.amodel.params += alpha * (self.env.gamma ** t ) * G * dparams

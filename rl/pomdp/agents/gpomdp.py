import logging
logger = logging.getLogger(__name__)

from .agent import Agent

import numpy as np


class GPOMDP(Agent):
    logger = logging.getLogger(f'{__name__}.GPOMDP')

    def reset(self):
        self.policy.reset()
        self.restart()

    def restart(self):
        self.policy.restart()
        self.ihistory = []

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        o = feedback.o

        icontext = self.policy.context
        ifeedback = self.policy.feedback(o)
        self.ihistory.append((icontext, ifeedback))

    # TODO implement step-variant!
    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        G = 0.

        history = zip(reversed(episode), reversed(self.ihistory))
        for (context, a, feedback), (icontext, ifeedback) in history:
            t = context.t
            r = feedback.r
            o = icontext.o

            G = r + self.env.gamma * G

            # TODO let some other method take care of how to do the gradient descent (normalized or not, etc...)
            alpha = .001
            dparams = self.policy.amodel.dlogprobs(o, a)
            # dparams /= np.sqrt((dparams * dparams).sum())
            self.policy.amodel.params += alpha * G * dparams
            # self.policy.amodel.params += alpha * (self.env.gamma ** t ) * G * dparams

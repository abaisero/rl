import logging
logger = logging.getLogger(__name__)

from .algo import Algo

import numpy as np


class PolicyGradient(Algo):
    logger = logging.getLogger(f'{__name__}.PolicyGradient')

    def __repr__(self):
        return repr(self.pgrad)

    def __init__(self, policy, pgrad, step_size=None):
    # def __init__(self, name, policy, pgrad, step_size=None):
        if step_size is None:
            step_size = optim.StepSize(1)

        # super().__init__(name, policy)
        self.policy = policy
        self.pgrad = pgrad
        self.step_size = step_size

    def reset(self):
        self.logger.debug('reset()')
        self.policy.reset()
        self.pgrad.reset()
        self.step_size.reset()

    def restart(self):
        self.pgrad.restart()

    def feedback(self, pcontext, context, a, feedback, context1):
        self.logger.info(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        try:
            pgrad_feedback = self.pgrad.feedback
        except AttributeError:
            self.policy.feedback(feedback)
            return

        dparams = pgrad_feedback(pcontext, context, a, feedback, context1)
        try:
            dparams *= self.step_size()
        except TypeError:
            return
        else:
            self.step_size.step()

        self.policy.params += dparams

    def feedback_episode(self, pcontext, episode):
        self.logger.info(f'feedback_episode() \t; len(episode)={len(episode)}')

        try:
            pgrad_feedback_episode = self.pgrad.feedback_episode
        except AttributeError:
            return

        dparams = pgrad_feedback_episode(pcontext, episode)

        try:
            dparams *= self.step_size()
        except TypeError:
            return
        else:
            self.step_size.step()

        lim = .1
        lim2 = lim ** 2

        gnorm2 = np.sum(_.sum() for _ in dparams ** 2)
        if gnorm2 > lim2:
            gnorm = np.sqrt(gnorm2)
            dparams *= lim / gnorm

        try:
            for cbe in self.callbacks_episode:
                cbe(dparams)
        except AttributeError:
            pass

        self.policy.params += dparams

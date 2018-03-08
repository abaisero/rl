import logging
logger = logging.getLogger(__name__)

from .agent import Agent

import numpy as np


class PolicyGradient(Agent):
    logger = logging.getLogger(f'{__name__}.PolicyGradient')

    def __init__(self, name, env, policy, pgrad, step_size=None):
        if step_size is None:
            step_size = optim.StepSize(1)

        super().__init__(name, env, policy)
        self.pgrad = pgrad
        self.step_size = step_size

    def reset(self):
        self.logger.debug('reset()')
        self.policy.reset()
        self.pgrad.reset()
        self.step_size.reset()

    def restart(self):
        self.logger.debug('restart()')
        self.policy.restart()
        self.pgrad.restart()

    def feedback(self, sys, context, a, feedback):
        self.logger.info(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        try:
            try:
                pgrad_feedback = self.pgrad.feedback
            except AttributeError:
                return

            dparams = pgrad_feedback(sys, context, a, feedback)
            try:
                dparams *= self.step_size()
            except TypeError:
                return
            else:
                self.step_size.step()

            self.policy.params += dparams
        finally:
            self.policy.feedback(feedback)

    def feedback_episode(self, sys, episode):
        self.logger.info(f'feedback_episode() \t; len(episode)={len(episode)}')

        try:
            pgrad_feedback_episode = self.pgrad.feedback_episode
        except AttributeError:
            return

        dparams = pgrad_feedback_episode(sys, episode)

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

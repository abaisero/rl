import logging
logger = logging.getLogger(__name__)

from .agent import Agent

import numpy as np


class PolicySearch(Agent):
    logger = logging.getLogger(f'{__name__}.PolicySearch')

    def __init__(self, name, env, policy, psearch):
        super().__init__(name, env, policy)
        self.psearch = psearch

    def reset(self):
        self.policy.reset()
        self.psearch.reset()

    def restart(self):
        self.policy.restart()
        self.psearch.restart()

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        try:
            try:
                psearch_feedback = self.psearch.feedback
            except AttributeError:
                return

            params = psearch_feedback(sys, context, a, feedback)
            if params is not None:
                self.policy.params = params
        finally:
            self.policy.feedback(feedback)

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        try:
            psearch_feedback_episode = self.psearch.feedback_episode
        except AttributeError:
            return

        params = psearch_feedback_episode(sys, episode)
        if params is not None:
            self.policy.params = params

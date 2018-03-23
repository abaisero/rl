import logging
logger = logging.getLogger(__name__)

from .algo import Algo

import numpy as np


class PolicySearch(Algo):
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

    def feedback(self, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        try:
            try:
                psearch_feedback = self.psearch.feedback
            except AttributeError:
                return

            params = psearch_feedback(context, a, feedback, context1)
            if params is not None:
                self.policy.params = params
        finally:
            self.policy.feedback(feedback)

    def feedback_episode(self, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        try:
            psearch_feedback_episode = self.psearch.feedback_episode
        except AttributeError:
            return

        params = psearch_feedback_episode(episode)
        if params is not None:
            self.policy.params = params

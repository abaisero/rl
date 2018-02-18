import logging
logger = logging.getLogger(__name__)

from .agent import Agent


class REINFORCE(Agent):
    logger = logging.getLogger(f'{__name__}.MonteCarloES')

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        G = 0.
        for context, a, feedback in reversed(episode):
            s = context.s
            r = feedback.r

            G = r + self.env.gamma * G
            # TODO how to handle alpha better?!
            alpha = .01
            self.policy.params += alpha * G * self.policy.dlogprobs(s, a)

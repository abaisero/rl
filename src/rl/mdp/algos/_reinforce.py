import logging
logger = logging.getLogger(__name__)

from .agent import Agent


class REINFORCE(Agent):
    logger = logging.getLogger(f'{__name__}.REINFORCE')

    def feedback(self, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t, s = context.t, context.s
        r, s1 = feedback.r, feedback.s1

        self.z = self.env.gamma * self.z + self.policy.dlogprobs(s, a)
        self.d += (r * self.z - self.d) / (t+1)

    def feedback_episode(self, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        alpha = 1
        self.policy.params += alpha * self.d


    # def feedback_episode(self, episode):
    #     self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

    #     G = 0.
    #     for context, a, feedback in reversed(episode):
    #         s = context.s
    #         r = feedback.r

    #         G = r + self.env.gamma * G
    #         # TODO how to handle alpha better?!
    #         alpha = .001
    #         dparams = self.policy.dlogprobs(s, a)

    #         self.policy.params += alpha * G * dparams

import logging
logger = logging.getLogger(__name__)

from .agent import Agent

import numpy.random as rnd


class Qlearning(Agent):
    logger = logging.getLogger(f'{__name__}.Qlearning')

    def __init__(self, name, env, policy, Q):
        super().__init__(name, env, policy)
        self.Q = Q
        self.counts = Q.counts()

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        s = context.s
        s1, r = feedback.s1, feedback.r

        target = r + self.env.gamma * self.Q.optim_value(s1)
        delta = target - self.Q[s, a]
        alpha = 1 / (self.counts[s, a] + 1)

        self.Q[s, a] += alpha * delta
        self.counts[s, a] += 1


class Qlearning_l(Agent):
    logger = logging.getLogger(f'{__name__}.Qlearning_l')

    def __init__(self, name, env, policy, Q, gamma, l):
        super().__init__(name, env, policy)
        self.Q = Q
        self.elig = Q.eligibility(gamma, l)

    def reset(self):
        self.elig.reset()

    def feedback(self,sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        s = context.s
        s1, r = feedback.s1, feedback.r

        self.elig[:] *= self.elig.gl
        self.elig[s, a] += 1

        target = r + self.env.gamma * self.Q.optim_value(s1)
        delta = target - self.Q[s, a]
        alpha = .1  # TODO how to specify alpha?

        self.Q[:] += alpha * delta * self.elig[:]


# class DoubleQlearning(Agent):
#     def __init__(self, name, env, policy, Q1, Q2):
#         super().__init__(name, env, policy)
#         self.Q1 = Q1
#         self.Q2 = Q2

#     def feedback(self, sys, context, a, feedback):
#         s = context.s
#         s1, r = feedback.s1, feedback.r

#         Qa, Qb = rnd.permutation((self.Q1, self.Q2))
#         amax = Qa.optim_action(s1)
#         target = r + self.env.gamma * Qb(s1, amax)
#         Qa.update_target(s, a, target)

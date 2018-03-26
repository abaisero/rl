import logging
logger = logging.getLogger(__name__)

from .agent import Agent, PreAgent
import rl.values.v as v


class SARSA(PreAgent):
    logger = logging.getLogger(f'{__name__}.SARSA')

    def __init__(self, name, env, policy, Q):
        super().__init__(name, env, policy)
        self.Q = Q
        self.counts = Q.counts()

    def feedback(self, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        s = context.s
        s1, r = feedback.s1, feedback.r
        a1 = self.act(context1, preact=True)

        target = r + self.env.gamma * self.Q[s1, a1]
        delta = target - self.Q[s, a]
        alpha = 1 / (self.counts[s, a] + 1)

        self.Q[s, a] += alpha * delta
        self.counts[s, a] += 1


class SARSA_l(PreAgent):
    logger = logging.getLogger(f'{__name__}.SARSA_l')

    def __init__(self, name, env, policy, Q, gamma, l):
        super().__init__(name, env, policy)
        self.Q = Q
        self.elig = Q.eligibility(gamma, l)

    def restart(self):
        # super().reset()
        self.elig.restart()

    def feedback(self, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        s = context.s
        s1, r = feedback.s1, feedback.r
        a1 = self.act(context1, preact=True)

        self.elig[:] *= self.elig.gl
        self.elig[s, a] += 1

        target = r + self.env.gamma * self.Q[s1, a1]
        delta = target - self.Q[s, a]
        alpha = .1  # TODO how to specify alpha?

        self.Q[:] += alpha * delta * self.elig[:]


class ExpectedSARSA(Agent):
    logger = logging.getLogger(f'{__name__}.ExpectedSARSA')

    def __init__(self, name, env, policy, Q):
        super().__init__(name, env, policy)
        self.Q = Q
        self.counts = Q.counts()
        self.V = v.QBased(Q, policy)

    def feedback(self, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        s = context.s
        s1, r = feedback.s1, feedback.r

        target = r + self.env.gamma * self.V(s1)
        delta = target - self.Q[s, a]
        alpha = 1 / (self.counts[s, a] + 1)

        self.Q[s, a] += alpha * delta
        self.counts[s, a] += 1

import logging
logger = logging.getLogger(__name__)

from .agent import Agent


# TODO not algorithms as individual agents; but as stuff the agent can do in
# between... this way we can run both evaluation and other stuff independently
# also this way the agent itself doesn't have the V, Q and other shit...

class TD(Agent):
    logger = logging.getLogger(f'{__name__}.TD')

    def __init__(self, name, env, policy, V):
        super().__init__(name, env, policy)
        self.V = V
        self.counts = V.counts()

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        s = context.s
        s1, r = feedback.s1, feedback.r

        target = r + self.env.gamma * self.V[s1]
        delta = target - self.V[s]
        alpha = 1 / (self.counts[s] + 1)

        self.V[s] += alpha * delta
        self.counts[s] += 1

        # self.V[s] += alpha * (target - self.V[s].update_target(s, target, alpha)
        # self.V.update_target(s, target, alpha)


        # TODO in the same vein, I want counts to be represented explicitly here...


class TD_l(Agent):
    logger = logging.getLogger(f'{__name__}.TD')

    # def __init__(self, name, env, policy, V, l):
    #     super().__init__(name, env, policy)
    #     self.V = V
    #     self.l = l

    # TODO I want the agent itself to have access to eligibility trace, and whatnot...  not the value function!!

    def __init__(self, name, env, policy, V, gamma, l):
        super().__init__(name, env, policy)
        self.V = V
        self.elig = V.eligibility(gamma, l)

    # def reset(self):
    #     self.V.reset()
    #     self.policy.reset()

    def restart(self):
        self.elig.restart()

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        s = context.s
        s1, r = feedback.s1, feedback.r

        # TODO this will have to change with linear value approximation..
        self.elig[s] += 1

        target = r + self.env.gamma * self.V[s1]
        delta = target - self.V[s]
        alpha = .1

        self.V[:] += alpha * delta * self.elig[:]
        self.elig[:] *= self.elig.gl

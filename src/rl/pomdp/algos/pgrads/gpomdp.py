import logging
logger = logging.getLogger(__name__)

from .p import P

import numpy as np

import argparse
from rl.misc.argparse import GroupedAction


class GPOMDP(P):
    """ "Infinite-Horizon Policy-Gradient Estimation" - J. Baxter, P.Bartlett """

    logger = logging.getLogger(f'{__name__}.GPOMDP')

    def __repr__(self):
        return f'GPOMDP(beta={self.beta})'

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--beta', type=float,
            required=False, default=None)

    @staticmethod
    def from_namespace(policy, namespace):
        return GPOMDP(policy, namespace.beta)


    def __init__(self, policy, beta):
        super().__init__(policy)
        self.beta = beta

    def restart(self):
        self.z = 0
        self.d = 0

    def feedback(self, pcontext, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t = context.t
        o = self.policy.context.o
        r = feedback.r

        self.z = self.beta * self.z + self.policy.dlogprobs(t, o, a)
        self.d += (r * self.z - self.d) / (t+1)

    def feedback_episode(self, pcontext, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        return self.d

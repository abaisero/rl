import logging
logger = logging.getLogger(__name__)

from .p import P

import numpy as np

import argparse
from rl.misc.argparse import GroupedAction


class IsGPOMDP(P):
    """ "Scaling Internal-State Policy-Gradient Methods for POMDPs" - D. Aberdeen, J. Baxter """

    logger = logging.getLogger(f'{__name__}.IsGPOMDP')

    def __repr__(self):
        return f'IsGPOMDP(beta={self.beta})'

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--beta', type=float,
            required=False, default=None)

    @staticmethod
    def from_namespace(policy, namespace):
        return IsGPOMDP(policy, namespace.beta)


    def __init__(self, policy, beta):
        super().__init__(policy)
        self.beta = beta

    def restart(self):
        self.z = 0
        self.d = 0

    def feedback(self, pcontext, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t = context.t
        o, r = feedback.o, feedback.r
        n = pcontext.n
        n1 = self.policy.sample_n(n, o)

        self.z = self.beta * self.z + self.policy.dlogprobs(n, a, o, n1)
        self.d += (r * self.z - self.d) / (t+1)

        pcontext.n = n1

    def feedback_episode(self, pcontext, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        return self.d

import logging
logger = logging.getLogger(__name__)

from .p import P

import rl.misc.models as models

import numpy as np

import argparse
from rl.misc.argparse import GroupedAction


class ExpGPOMDP(P):
    """ "..." """

    logger = logging.getLogger(f'{__name__}.ExpGPOMDP')

    def __repr__(self):
        return f'ExpGPOMDP(beta={self.beta})'

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--beta', type=float,
            required=False, default=None)

    @staticmethod
    def from_namespace(policy, namespace):
        return ExpGPOMDP(policy, namespace.beta)


    def __init__(self, policy, beta):
        super().__init__(policy)
        self.beta = beta

    def restart(self):
        self.z = 0
        self.d = 0

#     def act(self, context):
#         pass

    def dlogprobs(self, a, o):
        # TODO this is only initial;
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = np.zeros_like(self.policy.amodel.params)
        dlogprobs[1] = np.zeros_like(self.policy.nmodel.params)
        return dlogprobs

    def reset(self):
        self.db = np.zeros((self.policy.nnodes,) + self.policy.nmodel.dims)

    def feedback(self, pcontext, context, a, feedback, context1):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        t = context.t
        o, r = feedback.o, feedback.r
        b = pcontext.b

        # nabla mu
        dlogprobs = np.empty(2, dtype=object)

        _ = slice(None)
        dlogprobs[0] = np.tensordot(b.probs(_), self.policy.amodel.dprobs(_, a), 1) \
                / np.inner(b.probs(_), self.policy.amodel.probs(_, a))
        dlogprobs[1] = np.tensordot(self.policy.amodel.probs(_, a), self.db, 1) \
                / np.inner(b.probs(_), self.policy.amodel.probs(_, a))

        # gradient estimate updates
        self.z = self.beta * self.z + dlogprobs
        self.d += (r * self.z - self.d) / (t+1)

        # alpha updates
        b1 = b.probs(_) @ self.policy.nmodel.probs(_, o, _)
        # db1 = np.tensordot(self.policy.nmodel.probs(_, o, _), self.db, 1) \
        #     + np.tensordot(b.probs(_), self.policy.nmodel.dprobs(_, o, _), 1)
        # print(self.db.shape, self.policy.nmodel.probs(_, o, _).shape)
        db1 = np.tensordot(self.db, self.policy.nmodel.probs(_, o, _), 1) \
            + np.tensordot(b.probs(_), self.policy.nmodel.dprobs(_, o, _), 1)

        pcontext.b[:] = b1
        self.db = db1

    def feedback_episode(self, pcontext, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        return self.d

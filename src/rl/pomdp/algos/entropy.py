import types
import copy

import numpy as np


def entropy(params, policy, env):
    probs = policy.amodel.probs(params[0], ())
    dprobs = policy.amodel.dprobs(params[0], ())
    logprobs = policy.amodel.logprobs(params[0], ())

    c = np.einsum('na,na->', probs, logprobs)
    dc = np.zeros_like(params)
    dc[0] = np.einsum('na...,na->...', dprobs, -logprobs)
    dc[1] = np.zeros(1)

    return c, dc


class Entropy:
    type_ = 'episodic'

    def __init__(self, policy):
        self.policy = policy

    def new_context(self):
        return types.SimpleNamespace(
            obj=0.,  # objective value
            elig=0.,  # eligibility trace
            grad=0.,  # gradient
        )

    def step(self, params, acontext, econtext, pcontext, a, feedback,
             pcontext1, *, inline=False):
        if not inline:
            acontext = copy.copy(acontext)

        logprobs = self.policy.logprobs(params, pcontext, a, feedback,
                                        pcontext1)
        dlogprobs = self.policy.dlogprobs(params, pcontext, a, feedback,
                                          pcontext1)

        neglogp = -logprobs[0]
        acontext.obj += (neglogp - acontext.obj) / (econtext.t + 1)
        acontext.elig += dlogprobs
        acontext.grad += (neglogp * acontext.elig - acontext.grad) / \
                         (econtext.t + 1)

        return acontext

import types
import copy

import numpy as np


class Contextful:
    type_ = 'analytic'

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, params):
        N = self.policy.nnodes

        # TODO this is specific for amodel = params[0].  Generalize!
        probs = self.policy.amodel.probs(params[0], ())
        dprobs = self.policy.amodel.dprobs(params[0], ())
        logprobs = self.policy.amodel.logprobs(params[0], ())
        dlogprobs = self.policy.amodel.dlogprobs(params[0], ())

        dlogprobs_sum = dlogprobs.sum(axis=1, keepdims=True)
        logprobs_sum = logprobs.sum(axis=1, keepdims=True)

        c = np.einsum('na,na->', probs, N * logprobs - logprobs_sum)
        dc = np.zeros_like(params)
        dc[0] = np.einsum('na...,na->...',
                          dprobs, N * logprobs - logprobs_sum) \
            + np.einsum('na,na...->...', probs, N * dlogprobs - dlogprobs_sum)
        dc[1] = np.zeros(1)

        c /= N * (N - 1)
        dc /= N * (N - 1)

        return c, dc


class Contextful2:
    type_ = 'analytic'

    def __init__(self, policy, cf_probs):
        self.policy = policy
        self.cf_logprobs = np.log(cf_probs)

    def __call__(self, params):
        # TODO I need to know which params are relative to which part of stuff.

        # TODO this is specific for amodel = params[0].  Generalize!

        probs = self.policy.amodel.probs(params[0], ())
        dprobs = self.policy.amodel.dprobs(params[0], ())
        logprobs = self.policy.amodel.logprobs(params[0], ())

        logratio = logprobs - self.cf_logprobs
        obj = np.einsum('na,na->', probs, logratio)
        grad = np.zeros_like(params)
        grad[0] = np.einsum('na...,na->...', dprobs, logratio)
        grad[1] = np.zeros(1)

        obj /= self.policy.nnodes
        grad /= self.policy.nnodes

        obj = np.nan_to_num(obj)
        grad = np.nan_to_num(grad)

        return obj, grad


# def contextful3(params, policy, env, cfprobs, nsteps=100, beta=None):
#     if beta is None:
#         beta = env.gamma

#     logcfprobs = np.log(cfprobs)

#     KL = 0.
#     z, d = 0, 0

#     econtext = env.new_context()
#     pcontext = policy.new_context(params)
#     while econtext.t < nsteps:
#         a = policy.sample_a(params, pcontext)
#         feedback, econtext = env.step(econtext, a)
#         # NOTE econtext already takes step here
#         pcontext1 = policy.step(params, pcontext, feedback, inline=False)

#         logprobs = policy.logprobs(params, pcontext, a, feedback, pcontext1)
#         dlogprobs = policy.dlogprobs(params, pcontext, a, feedback,
#                                      pcontext1)

#         KL += (logprobs[0] - logcfprobs[a] - KL) / econtext.t

#         z += beta * z + dlogprobs
#         d += ((logprobs - logcfprobs[a]) * z - d) / econtext.t
#         pcontext = pcontext1

#     return KL, d


class Contextful3:
    type_ = 'episodic'

    def __init__(self, policy, beta, cf_probs):
        self.policy = policy
        self.beta = beta
        self.cf_logprobs = np.log(cf_probs)

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

        logratio = logprobs - self.cf_logprobs[a]
        acontext.obj += (logratio[0] - acontext.obj) / (econtext.t + 1)
        acontext.elig = self.beta * acontext.elig + dlogprobs
        acontext.grad += (logratio * acontext.elig - acontext.grad) / \
                         (econtext.t + 1)

        return acontext

    def episode(self, params, policy, env, nsteps):
        econtext = env.new_context()
        pcontext = policy.new_context(params)
        acontext = self.new_context()
        while econtext.t < nsteps:
            a = policy.sample_a(params, pcontext)
            feedback, econtext1 = env.step(econtext, a)
            pcontext1 = policy.step(params, pcontext, feedback)
            pscore = policy.dlogprobs(params, pcontext, a, feedback, pcontext1)

            self.step(params, acontext, econtext, feedback, pscore,
                      inline=True)

            econtext = econtext1
            pcontext = pcontext1

        return acontext

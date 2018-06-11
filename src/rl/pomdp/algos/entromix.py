import types
import copy

import numpy as np


class Entromix:
    type_ = 'episodic'

    def __init__(self, policy, cf_probs):
        self.policy = policy
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

        dlogprobs = self.policy.dlogprobs(params, pcontext, a, feedback,
                                          pcontext1)

        logratio = -self.cf_logprobs[a]
        acontext.obj += (logratio - acontext.obj) / (econtext.t + 1)
        acontext.elig += dlogprobs
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

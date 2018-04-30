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

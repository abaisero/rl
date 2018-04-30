import numpy as np
# from scipy.stats import entropy


def contextful(params, policy, env, l=None):
    N = policy.nnodes

    # TODO this is specific for amodel = params[0].  Generalize!
    probs = policy.amodel.probs(params[0], ())
    dprobs = policy.amodel.dprobs(params[0], ())
    logprobs = policy.amodel.logprobs(params[0], ())
    dlogprobs = policy.amodel.dlogprobs(params[0], ())

    dlogprobs_sum = dlogprobs.sum(axis=1, keepdims=True)
    logprobs_sum = logprobs.sum(axis=1, keepdims=True)

    c = np.einsum('na,na->', probs, N * logprobs - logprobs_sum)
    dc = np.zeros_like(params)
    dc[0] = np.einsum('na...,na->...', dprobs, N * logprobs - logprobs_sum) \
        + np.einsum('na,na...->...', probs, N * dlogprobs - dlogprobs_sum)
    dc[1] = np.zeros(1)

    c /= N * (N - 1)
    dc /= N * (N - 1)

    if l is not None:
        softmax = np.logaddexp(l, c)
        c += l - softmax
        dc *= np.exp(l - softmax)

    return c, dc


# NOTE this is only for FSC
# how to generalize?
def contextful2(params, policy, env, cfprobs, l=None):
    N = policy.nnodes

    # TODO I need to know which params are relative to which part of stuff..

    # TODO this is specific for amodel = params[0].  Generalize!
    logcfprobs = np.nan_to_num(np.log(cfprobs))

    probs = policy.amodel.probs(params[0], ())
    dprobs = policy.amodel.dprobs(params[0], ())
    logprobs = policy.amodel.logprobs(params[0], ())

    # import ipdb
    # ipdb.set_trace()
    c = np.einsum('na,na->', probs, logprobs - logcfprobs)
    dc = np.zeros_like(params)
    dc[0] = np.einsum('na...,na->...', dprobs, logprobs - logcfprobs)
    dc[1] = np.zeros(1)

    c /= N
    dc /= N

    # TODO compute this better!
    if l is not None:
        softmax = np.logaddexp(l, c)
        c += l - softmax
        dc *= np.exp(l - softmax)

    c = np.nan_to_num(c)
    dc = np.nan_to_num(dc)

    # print(c)
    return c, dc


def contextful3(params, policy, env, cfprobs, nsteps=100, beta=None, l=None):
    if beta is None:
        beta = env.gamma

    logcfprobs = np.log(cfprobs)

    KL = 0.
    z, d = 0, 0

    econtext = env.new_context()
    pcontext = policy.new_context(params)
    while econtext.t < nsteps:
        a = policy.sample_a(params, pcontext)
        feedback, econtext = env.step(econtext, a)
        # NOTE econtext already takes step here
        pcontext1 = policy.step(params, pcontext, feedback, inline=False)

        logprobs = policy.logprobs(params, pcontext, a, feedback, pcontext1)
        dlogprobs = policy.dlogprobs(params, pcontext, a, feedback, pcontext1)

        KL += (logprobs[0] - logcfprobs[a] - KL) / econtext.t

        z += beta * z + dlogprobs
        d += ((logprobs - logcfprobs[a]) * z - d) / econtext.t
        pcontext = pcontext1

    KL /= nsteps
    d /= nsteps

    if l is not None:
        softmax = np.logaddexp(l, KL)
        KL += l - softmax
        d *= np.exp(l - softmax)

    return KL, d

import numpy as np


def softmin(params, policy, env, algo, l):
    v, g = algo(params, policy, env)

    softmax = np.logaddexp(l, v)
    v += l - softmax
    g *= np.exp(l - softmax)

    return v, g

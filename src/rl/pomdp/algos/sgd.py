def pgradient(policy, pgrad, niter):
    for i in range(niter):
        dparams = pgrad()
        for p, dp in zip(policy.params, dparams):
            p[:] += dp


def pgradient(params, gradient, niter):
    for i in range(niter):
        dparams = gradient()
        for p, dp in zip(params, dparams):
            p[:] += dp





def SGD(params, gradient, niter):
    for i in range(niter):
        params += gradient(params)

def pSGC(policy, gradient, niter):
    for i in range(niter):
        policy.params += gradient(policy)


def IsGPOMDP(policy, env, beta):
    env.reset()
    while not env.done:
        t = env.t
        n = pcontext.n
        a = policy.sample(pcontext)
        r, o = env.step(a)
        n1 = policy.sample_n(n, o)

        z = beta * z + policy.dlogprobs(n, a, o, n1)
        d += (r * z - d) / (t+1)

        pcontext.n = n1
    return d


def ContextFreeCost(policy, l):
    D = 0.
    for n in policy.nodes:
        pa = policy.amodel.dist(n)
        for n_ in policy.nodes:
            if n != n_:
                pa_ = policy.amodel.dist(n_)
                D += entropy(pa, pa_)

    return l * D


def ContextFreeGradient(policy, l):
    probs = policy.amodel.probs()
    logprobs = policy.amodel.logprobs()
    dlogprobs = policy.amodel.dlogprobs()

    A = np.einsum('ia,ia...,ia->...',
        probs, dlogprobs,
        N * logprobs - logprobs.sum(axis=0, keepdims=True)
    )
    B = np.einsum('ia,ia...->...',
        probs,
        N * dlogprobs - dlogprobs.sum(axis=0, keepdims=True)
    )

    return l * (A + B)

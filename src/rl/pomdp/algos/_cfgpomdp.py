import argparse
import logging
logger = logging.getLogger(__name__)

from .algo import Algo


class CFGPOMDP(Algo):
    """ "..." """

    logger = logging.getLogger(f'{__name__}.CFGPOMDP')

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __repr__(self):
        return f'CFGPOMDP(beta={self.beta})'

    def episode(self, env, policy, nsteps):
        env.reset()

        z, d = 0, 0

        pcontext = policy.new_pcontext()
        while env.t < nsteps:
            t = env.t
            a = policy.sample(pcontext)
            r, _ = env.step(a)

            z = self.beta * z + policy.dlogprobs(a)
            d += (r * z - d) / (t+1)
        return d


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--beta', type=float,
            required=False, default=None)

    @staticmethod
    def from_namespace(namespace):
        return CFGPOMDP(namespace.beta)


def cfgpomdp(params, policy, env, nsteps=100, beta=None):
    if beta is None:
        beta = env.gamma

    g = 0.
    z, d = 0, 0

    econtext = env.new_context()
    pcontext = policy.new_pcontext()
    while econtext.t < nsteps:
        a = policy.sample_a(params, pcontext)
        feedback, econtext = env.step(econtext, a)
        pcontext1 = policy.step(params, pcontext, feedback, inline=False)

        z = beta * z + policy.dlogprobs(params, pcontext, a, feedback, pcontext1)
        d += (feedback.r * z - d) / (t+1)
        pcontext = pcontext1

    return g, d

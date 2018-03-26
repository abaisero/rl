import argparse
import logging
logger = logging.getLogger(__name__)

from .algo import Algo


class IsGPOMDP(Algo):
    """ "Scaling Internal-State Policy-Gradient Methods for POMDPs" - D. Aberdeen, J. Baxter """

    logger = logging.getLogger(f'{__name__}.IsGPOMDP')

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __repr__(self):
        return f'IsGPOMDP(beta={self.beta})'

    def episode(self, env, policy, nsteps):
        env.reset()

        z, d = 0, 0

        pcontext = policy.new_pcontext()
        while env.t < nsteps:
            t = env.t
            n = pcontext.n
            a = policy.sample(pcontext)
            r, o = env.step(a)
            n1 = policy.sample_n(n, o)

            z = self.beta * z + policy.dlogprobs(n, a, o, n1)
            d += (r * z - d) / (t+1)

            pcontext.n = n1
        return d


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--beta', type=float,
            required=False, default=None)

    @staticmethod
    def from_namespace(namespace):
        return IsGPOMDP(namespace.beta)

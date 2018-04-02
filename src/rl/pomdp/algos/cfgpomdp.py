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

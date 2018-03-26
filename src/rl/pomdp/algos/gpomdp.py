import argparse
import logging
logger = logging.getLogger(__name__)

from .algo import Algo


class GPOMDP(Algo):
    """ "Infinite-Horizon Policy-Gradient Estimation" - J. Baxter, P.Bartlett """

    logger = logging.getLogger(f'{__name__}.GPOMDP')

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __repr__(self):
        return f'GPOMDP(beta={self.beta})'

    def episode(self, env, policy, nsteps):
        raise NotImplementedError


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--beta', type=float,
            required=False, default=None)

    @staticmethod
    def from_namespace(namespace):
        return GPOMDP(namespace.beta)

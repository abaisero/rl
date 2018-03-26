import argparse
import logging
logger = logging.getLogger(__name__)

from .algo import Algo

import numpy as np


class ExpGPOMDP(Algo):
    """ "Scaling Internal-State Policy-Gradient Methods for POMDPs" - D. Aberdeen, J. Baxter """

    logger = logging.getLogger(f'{__name__}.IsGPOMDP')

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __repr__(self):
        return f'ExpGPOMDP(beta={self.beta})'

    def episode(self, env, policy, nsteps):
        env.reset()

        db = np.zeros((policy.nnodes,) + policy.nmodel.dims)
        dlogprobs = np.empty(2, dtype=object)
        z, d = 0, 0
        _ = slice(None)

        pcontext = policy.new_pcontext()
        while env.t < nsteps:
            t = env.t
            b = pcontext.b
            a = policy.sample(pcontext)
            r, o = env.step(a)

            dlogprobs[0] = np.tensordot(b.probs(_), policy.amodel.dprobs(_, a), 1) \
                    / np.inner(b.probs(_), policy.amodel.probs(_, a))
            dlogprobs[1] = np.tensordot(policy.amodel.probs(_, a), db, 1) \
                    / np.inner(b.probs(_), policy.amodel.probs(_, a))

            z = self.beta * z + dlogprobs
            d += (r * z - d) / (t+1)

            b1 = b.probs(_) @ policy.nmodel.probs(_, o, _)
            db1 = np.tensordot(db, policy.nmodel.probs(_, o, _), 1) \
                + np.tensordot(b.probs(_), policy.nmodel.dprobs(_, o, _), 1)

            b[:] = b1
        return d


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--beta', type=float,
            required=False, default=None)

    @staticmethod
    def from_namespace(namespace):
        return ExpGPOMDP(namespace.beta)

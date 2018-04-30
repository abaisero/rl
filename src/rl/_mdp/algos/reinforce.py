import argparse
import logging
logger = logging.getLogger(__name__)

from .algo import Algo


class REINFORCE(Algo):
    """ "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" - R. J. Williams """

    logger = logging.getLogger(f'{__name__}.REINFORCE')

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f'REINFORCE()'

    def episode(self, env, policy, nsteps):
        env.reset()

        z, d = 0, 0

        pcontext = policy.new_pcontext()
        pcontext.s = env.s
        while env.t < nsteps:
            t, s = env.t, env.s
            a = policy.sample(pcontext)
            r, s1 = env.step(a)

            z = env.mdp.gamma * z + policy.dlogprobs(s, a)
            d += (r * z - d) / (t+1)

            pcontext.s = s1
        return d

    parser = argparse.ArgumentParser(add_help=False)

    @staticmethod
    def from_namespace(namespace):
        return REINFORCE()

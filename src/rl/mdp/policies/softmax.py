from .policy import Policy

import argparse
from types import SimpleNamespace

import rl.misc.models as models


class Softmax(Policy):
    def __init__(self, mdp):
        self.amodel = models.Softmax(mdp.aspace, cond=(mdp.sspace,))

    def __repr__(self):
        return f'Softmax()'

    @property
    def params(self):
        return self.amodel.params

    @params.setter
    def params(self, value):
        self.amodel.params = value

    def dlogprobs(self, s, a):
        return self.amodel.dlogprobs(s, a)

    def new_pcontext(self):
        return SimpleNamespace()

    def reset(self):
        self.amodel.reset()

    def dist(self, pcontext):
        return self.amodel.dist(pcontext.s)

    def pr(self, pcontext, a):
        return self.amodel.pr(pcontext.s, a)

    def sample(self, pcontext):
        return self.amodel.sample(pcontext.s)

    parser = argparse.ArgumentParser(add_help=False)

    @staticmethod
    def from_namespace(mdp, namespace):
        return Softmax(mdp)

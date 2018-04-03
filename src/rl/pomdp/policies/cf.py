from .policy import Policy
import rl.graph as graph

import argparse
from types import SimpleNamespace

import rl.misc.models as models


class CF(Policy):
    def __init__(self, pomdp):
        self.amodel = models.Softmax(pomdp.aspace)

    def __repr__(self):
        return f'CF()'

    @property
    def params(self):
        return self.amodel.params

    @params.setter
    def params(self, value):
        self.amodel.params = value

    def reset(self):
        self.amodel.reset()

    def new_pcontext(self):
        return SimpleNamespace()

    def dlogprobs(self, a):
        return self.amodel.dlogprobs(a)

    def dist(self, pcontext):
        return self.amodel.dist()

    def pr(self, pcontext, a):
        return self.amodel.pr(a)

    def sample(self, pcontext):
        return self.amodel.sample()

    def plot(self, pomdp, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.cfplot(self, pomdp, nepisodes)
        self.idx = 0

    def plot_update(self):
        adist = self.amodel.probs()

        self.q.put((self.idx, adist))
        self.idx += 1

        if self.idx == self.neps:
            self.q.put(None)


    parser = argparse.ArgumentParser(add_help=False)

    @staticmethod
    def from_namespace(pomdp, namespace):
        return CF(pomdp)

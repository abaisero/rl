from .policy import Policy
import rl.graph as graph

# import argparse
# from rl.misc.argparse import GroupedAction

import indextools
import rl.misc.models as models

from collections import namedtuple
from types import SimpleNamespace

import numpy as np


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')

# PContext = SimpleNamespace('PContext', 'amodel, nmodel, n')


class BeliefFSC(Policy):
    def __init__(self, env, fsc):
        super().__init__(env)
        self.fsc = fsc

    def __repr__(self):
        return f'Belief-{self.fsc}'

    @property
    def params(self):
        return self.fsc.params

    @params.setter
    def params(self, value):
        self.fsc.params = value

    def dlogprobs(self, n, a, o, n1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(n, a)
        dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
        return dlogprobs

    def new_pcontext(self):
        b = models.Tabular(self.fsc.nspace)
        return SimpleNamespace(b=b)

    def reset(self):
        self.fsc.reset()

    def restart(self):
        pass

    @property
    def nspace(self):
        return self.fsc.nspace

    @property
    def nodes(self):
        return self.nspace.elems

    @property
    def nnodes(self):
        return self.nspace.nelems

    # @property
    # def context(self):
    #     pass
    #     # return IContext(self.n)

    # def feedback(self, feedback):
    #     pass
    #     # return self.feedback_o(feedback.o)

    # def feedback_o(self, o):
    #     pass
    #     # self.n = self.nmodel.sample(self.n, o)
    #     # return IFeedback(n1=self.n)

    @property
    def amodel(self):
        return self.fsc.amodel

    @property
    def nmodel(self):
        return self.fsc.nmodel

    def dist(self, pcontext):
        raise NotImplementedError
        # return self.amodel.dist(self.n)
        # return pcontext.b. self.amodel.dist(pcontext.n)

    def pr(self, pcontext, a):
        raise NotImplementedError
        # return self.amodel.pr(self.n, a)
        return self.amodel.pr(pcontext.n, a)

    def sample(self, pcontext):
        pcontext.n = pcontext.b.sample()
        return self.fsc.sample(pcontext)

    # def sample_n(self, n, o):
    #     return self.nmodel.sample(n, o)


    def plot(self, pomdp, nepisodes):
        self.fsc.plot(pomdp, nepisodes)
        # raise NotImplementedError
        # self.neps = nepisodes
        # self.q, self.p = graph.fscplot(self, nepisodes)
        # self.idx = 0

    def plot_update(self):
        self.fsc.plot_update()
        # raise NotImplementedError
        # adist = self.amodel.probs()
        # adist /= adist.sum(axis=-1, keepdims=True)

        # ndist = self.nmodel.probs()
        # ndist /= ndist.sum(axis=-1, keepdims=True)

        # self.q.put((self.idx, adist, ndist))
        # self.idx += 1

        # if self.idx == self.neps:
        #     self.q.put(None)

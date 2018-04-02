from .policy import Policy
import rl.graph as graph

import argparse
from types import SimpleNamespace

import indextools
import rl.misc.models as models

import numpy as np


class FSC(Policy):
    def __init__(self, pomdp, nspace):
        self.nspace = nspace

        self.amodel = models.Softmax(pomdp.aspace, cond=(nspace,))
        self.nmodel = models.Softmax(nspace, cond=(nspace, pomdp.ospace))

    def __repr__(self):
        return f'FSC(|N|={self.nnodes})'

    @property
    def params(self):
        params = np.empty(2, dtype=object)
        params[:] = self.amodel.params, self.nmodel.params
        return params

    @params.setter
    def params(self, value):
        aparams, nparams = value
        self.amodel.params = aparams
        self.nmodel.params = nparams

    def dlogprobs(self, n, a, o, n1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(n, a)
        dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
        return dlogprobs

    def new_pcontext(self):
        n = self.nspace.elem(0)
        return SimpleNamespace(n=n)

    def reset(self):
        self.amodel.reset()
        self.nmodel.reset()

    @property
    def nodes(self):
        return self.nspace.elems

    @property
    def nnodes(self):
        return self.nspace.nelems

    def dist(self, pcontext):
        return self.amodel.dist(pcontext.n)

    def pr(self, pcontext, a):
        return self.amodel.pr(pcontext.n, a)

    def sample(self, pcontext):
        return self.amodel.sample(pcontext.n)

    def dist_n(self, n, o):
        return self.nmodel.dist(n, o)

    def pr_n(self, n, o, n1):
        return self.nmodel.pr(n, o, n1)

    def sample_n(self, n, o):
        return self.nmodel.sample(n, o)

    def plot(self, pomdp, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.fscplot(self, pomdp, nepisodes)
        self.idx = 0

    def plot_update(self):
        adist = self.amodel.probs()
        adist /= adist.sum(axis=-1, keepdims=True)

        ndist = self.nmodel.probs()
        ndist /= ndist.sum(axis=-1, keepdims=True)

        self.q.put((self.idx, adist, ndist))
        self.idx += 1

        if self.idx == self.neps:
            self.q.put(None)


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('n', type=int)

    parser.add_argument('--belief', action='store_const', const=True,
            default=False)

    @staticmethod
    def from_namespace(pomdp, namespace):
        nodes = [f'node_{i}' for i in range(namespace.n)]
        nspace = indextools.DomainSpace(nodes)

        return FSC(pomdp, nspace)

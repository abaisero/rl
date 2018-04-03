from .policy import Policy
import rl.graph as graph

import argparse
from rl.misc.argparse import GroupedAction

import indextools
import rl.misc.models as models

from collections import namedtuple
from types import SimpleNamespace
import itertools as itt

import numpy as np
import numpy.linalg as la
import numpy.random as rnd


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')

class SparseFSC(Policy):
    def __init__(self, pomdp, nspace, K):
        super().__init__(pomdp)
        N = nspace.nelems
        O = pomdp.nobs

        self.nspace = nspace
        self.N = N
        self.K = K

        _combs = list(itt.combinations(range(O), 2))
        _test_mask = np.zeros((O, len(_combs)))
        for i, comb in enumerate(_combs):
            _test_mask[comb, i] = 1, -1

        for nfails in itt.count():
            if nfails == 100:
                raise ValueError(f'Could not initialize {self}')

            nmask = np.array([[
                    rnd.permutation(N)
                for _ in range(O)]
                for _ in range(N)]) < K

            # check that graph is not disjoint
            _nn = nmask.sum(axis=1)
            test = la.multi_dot([_nn] * N)
            if np.any(test == 0):
                continue

            # check that each observation gives a different transition mask
            test = np.einsum('hyg,yn->hng', nmask, _test_mask)
            if np.all(test == 0, axis=0).any():
                continue

            break

        self.amodel = models.Softmax(pomdp.aspace, cond=(nspace,))
        self.nmodel = models.Softmax(nspace, cond=(nspace, pomdp.ospace), mask=nmask)
        self.nmask = nmask.T


    def __repr__(self):
        return f'FSC_Sparse(N={self.N}, K={self.K})'

    @property
    def params(self):
        # TODO need better way to handle multiparametric models...
        # maybe just concatenate?  seems wrong..
        params = np.empty(2, dtype=object)
        params[:] = self.amodel.params, self.nmodel.params
        return params

    @params.setter
    def params(self, value):
        aparams, oparams = value
        self.amodel.params = aparams
        self.nmodel.params = oparams

    # def nk2n(self, n, k):
    #     n1idx = self.nkn[:, k, n].nonzero()[0].item()
    #     return self.nspace.elem(n1idx)

    # def nn2k(self, n, n1):
    #     k1idx = self.nkn[n1, :, n].nonzero()[0].item()
    #     return self.kspace.elem(k1idx)

    def dlogprobs(self, n, a, o, n1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(n, a)
        dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
        return dlogprobs

    def new_pcontext(self):
        # TODO belief over initial node as well...?
        n = self.nspace.elem(0)
        return SimpleNamespace(n=n)

    def reset(self):
        self.amodel.reset()
        self.nmodel.reset()

    @property
    def nodes(self):
        return self.nspace.nelems

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
    #     # k = self.kmodel.sample(self.n, o)
    #     # self.n = self.nk2n(self.n, k)
    #     # return IFeedback(n1=self.n)

    def dist(self, pcontext):
        # return self.amodel.dist(self.n)
        return self.amodel.dist(pcontext.n)

    def pr(self, pcontext, a):
        # return self.amodel.pr(self.n, a)
        return self.amodel.pr(pcontext.n, a)

    def sample(self, pcontext):
        # return self.amodel.sample(self.n)
        return self.amodel.sample(pcontext.n)

    def sample_n(self, n, o):
        return self.nmodel.sample(n, o)

    def plot(self, pomdp, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.sparsefscplot(self, pomdp, nepisodes)
        self.idx = 0

    def plot_update(self):
        adist = self.amodel.probs()
        ndist = self.nmodel.probs()

        self.q.put((self.idx, adist, ndist))
        self.idx += 1

        if self.idx == self.neps:
            self.q.put(None)

    # @staticmethod
    # def parser(group=None):
    #     def group_fmt(dest):
    #         return dest if group is None else f'{group}.{dest}'

    #     parser = argparse.ArgumentParser(add_help=False)
    #     parser.add_argument(dest=group_fmt('n'), metavar='n', type=int,
    #             action=GroupedAction, default=argparse.SUPPRESS)
    #     parser.add_argument(dest=group_fmt('k'), metavar='k', type=int,
    #             action=GroupedAction, default=argparse.SUPPRESS)

    #     parser.add_argument('--belief', action='store_const', const=True,
    #             default=False)

    #     return parser

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)

    parser.add_argument('--belief', action='store_const', const=True,
            default=False)

    @staticmethod
    def from_namespace(env, namespace):
        nodes = [f'node_{i}' for i in range(namespace.n)]
        nspace = indextools.DomainSpace(nodes)

        return SparseFSC(env, nspace, namespace.k)

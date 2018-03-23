from .policy import Policy
import rl.graph as graph

import argparse
from rl.misc.argparse import GroupedAction

import indextools
import rl.misc.models as models

from collections import namedtuple
from types import SimpleNamespace

import numpy as np


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')

# PContext = SimpleNamespace('PContext', 'amodel, nmodel, n')


# @add_subparser('fsc')
class FSC(Policy):
    def __repr__(self):
        return f'FSC(N={self.N})'

    def __init__(self, env, N):
        super().__init__(env)
        self.N = N  # number of nodes

        nodes = [f'node_{i}' for i in range(N)]
        self.nspace = indextools.DomainSpace(nodes)

        self.amodel = models.Softmax(env.aspace, cond=(self.nspace,))
        self.nmodel = models.Softmax(self.nspace, cond=(self.nspace, env.ospace))

        # TODO look at pgradient;  this won't work for some reason
        # self.params = np.array([self.amodel.params, self.nmodel.params])

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

    # def restart(self):
    #     pass
    #     # self.n = self.nspace.elem(0)

    @property
    def nodes(self):
        return self.nspace.elems

    @property
    def nnodes(self):
        return self.nspace.nelems

    # def feedback(self, feedback):
    #     pass
    #     # return self.feedback_o(feedback.o)

    # def feedback_o(self, o):
    #     pass
    #     # self.n = self.nmodel.sample(self.n, o)
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

    def plot(self, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.fscplot(self, nepisodes)
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


    # @staticmethod
    # def parser(group=None):
    #     def group_fmt(dest):
    #         return dest if group is None else f'{group}.{dest}'

    #     parser = argparse.ArgumentParser(add_help=False)
    #     parser.add_argument(dest=group_fmt('n'), metavar='n', type=int,
    #             action=GroupedAction, default=argparse.SUPPRESS)

    #     parser.add_argument('--belief', action='store_const', const=True,
    #             default=False)

    #     return parser

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('n', type=int)

    parser.add_argument('--belief', action='store_const', const=True,
            default=False)

    @staticmethod
    def from_namespace(env, namespace):
        return FSC(env, namespace.n)

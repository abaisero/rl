from .policy import Policy
import rl.graph as graph

import argparse
from rl.misc.argparse import GroupedAction

import pytk.factory as factory
import pytk.factory.model as fmodel

from collections import namedtuple
import numpy as np


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')


class FSC(Policy):
    def __init__(self, env, N, consistent=False):
        super().__init__(env)
        self.N = N  # number of nodes
        self.consistent = consistent

        values = [f'node_{i}' for i in range(N)]
        self.nfactory = factory.FactoryValues(values)

        self.amodel = fmodel.Softmax(env.afactory, cond=(self.nfactory,))
        if consistent:
            self.nmodel = fmodel.Softmax(self.nfactory, cond=(self.nfactory, env.afactory, env.ofactory))
        else:
            self.nmodel = fmodel.Softmax(self.nfactory, cond=(self.nfactory, env.ofactory))

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
        if self.consistent:
            dlogprobs[1] = self.nmodel.dlogprobs(n, a, o, n1)
        else:
            dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
        return dlogprobs

    def reset(self):
        self.amodel.reset()
        self.nmodel.reset()

    def restart(self):
        self.n = self.nfactory.item(0)

    @property
    def nodes(self):
        return self.nfactory.items

    @property
    def nnodes(self):
        return self.nfactory.nitems

    @property
    def context(self):
        return IContext(self.n)

    def feedback(self, feedback):
        return self.feedback_o(feedback.o)

    def feedback_o(self, o):
        if self.consistent:
            self.n = self.nmodel.sample(self.n, self.a, o)
        else:
            self.n = self.nmodel.sample(self.n, o)
        return IFeedback(n1=self.n)

    def dist(self):
        return self.amodel.dist(self.n)

    def pr(self, a):
        return self.amodel.pr(self.n, a)

    def sample(self):
        self.a = self.amodel.sample(self.n)
        return self.a

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


    @staticmethod
    def parser(group=None):
        def group_fmt(dest):
            return dest if group is None else f'{group}.{dest}'

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(dest=group_fmt('n'), metavar='n', type=int, action=GroupedAction, default=argparse.SUPPRESS)
        return parser

    @staticmethod
    def from_namespace(env, namespace):
        return FSC(env, namespace.n)

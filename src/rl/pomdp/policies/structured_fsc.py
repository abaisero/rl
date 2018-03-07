from .policy import Policy
import rl.graph as graph

import pytk.factory as factory
import pytk.factory.model as fmodel

from collections import namedtuple
import numpy as np
import numpy.linalg as la
import numpy.random as rnd


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')


class StructuredFSC(Policy):
    def __init__(self, env, amask, nmask, n0):
        super().__init__(env)

        if not amask.shape[0] == env.nactions:
            raise ValueError(f'Action mask shape {amask.shape} is wrong.')
        if not amask.shape[1] == nmask.shape[0] == nmask.shape[1]:
            raise ValueError(f'Action mask shape {amask.shape} and/or node mask shape {nmask.shape} is wrong.')

        self.amask = amask
        self.nmask = nmask
        self.N = amask.shape[1]

        # TODO get nfactory from outside, even?
        # TODO I want to get these directly from the outside...
        values = [f'node_{i}' for i in range(self.N)]
        # self.nfactory = factory.FactoryN(N)
        self.nfactory = factory.FactoryValues(values)
        self.n0 = self.nfactory.item(n0)

        self.amodel = fmodel.Softmax(env.afactory, cond=(self.nfactory,))
        self.nmodel = fmodel.Softmax(self.nfactory, cond=(self.nfactory, env.ofactory))
        # TODO I think I want this sparsity to happen directly in a sparse fmodel

        # TODO look at pgradient;  this won't work for some reason
        # self.params = np.array([self.amodel.params, self.nmodel.params])

    @property
    def params(self):
        # TODO need better way to handle multiparametric models...
        # maybe just concatenate?  seems wrong..
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

    def reset(self):
        self.amodel.reset()
        self.nmodel.reset()
        self.amodel.params[~self.amask.T] = -np.inf
        nmask = np.stack([self.nmask] * self.env.nobs, axis=1)
        self.nmodel.params[~nmask.T] = -np.inf

    def restart(self):
        self.n = self.n0.copy()

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
        self.q, self.p = graph.structuredfscplot(self, nepisodes)
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

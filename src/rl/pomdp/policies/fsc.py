from .policy import Policy

import pytk.factory as factory
import pytk.factory.model as fmodel

from collections import namedtuple
import numpy as np


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')


# TODO reset!!!!! choose initial state how? just first one?

class FSC(Policy):
    def __init__(self, env, N):
        super().__init__(env)
        self.N = N  # number of nodes
        self.nfactory = factory.FactoryN(N)

        self.amodel = fmodel.Softmax(env.afactory, cond=(self.nfactory,))
        self.omodel = fmodel.Softmax(self.nfactory, cond=(self.nfactory, env.ofactory))
        # TODO look at pgradient;  this won't work for some reason
        # self.params = np.array([self.amodel.params, self.omodel.params])

    @property
    def params(self):
        params = np.empty(2, dtype=object)
        params[:] = self.amodel.params, self.omodel.params
        return params

    @params.setter
    def params(self, value):
        aparams, oparams = value
        self.amodel.params = aparams
        self.omodel.params = oparams

    def reset(self):
        self.amodel.reset()
        self.omodel.reset()

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
        self.n = self.omodel.sample(self.n, o)
        return IFeedback(n1=self.n)

    def dist(self):
        return self.amodel.dist(self.n)

    def pr(self, a):
        return self.amodel.pr(self.n, a)

    def sample(self):
        return self.amodel.sample(self.n)

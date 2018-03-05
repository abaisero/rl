from .policy import Policy
import rl.graph as graph

import pytk.factory.model as fmodel

from collections import namedtuple


IContext = namedtuple('IContext', 'o')
IFeedback = namedtuple('IFeedback', 'o1')

# TODO reset!!!!! how?! does one receive an observation to begin with?
import numpy as np


class Reactive(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.a0model = fmodel.Softmax(env.afactory)
        self.amodel = fmodel.Softmax(env.afactory, cond=(env.ofactory,))
        # TODO look at pgradient;  this won't work for some reason
        # self.params = self.amodel.params

    @property
    def params(self):
        params = np.empty(2, dtype=object)
        params[:] = self.a0model.params, self.amodel.params
        return params

    @params.setter
    def params(self, value):
        a0params, aparams = value
        self.a0model.params = a0params
        self.amodel.params = aparams

    def reset(self):
        self.a0model.reset()
        self.amodel.reset()

    def restart(self):
        self.o = None

    @property
    def context(self):
        return IContext(self.o)

    def feedback(self, feedback):
        return self.feedback_o(feedback.o)

    def feedback_o(self, o):
        self.o = o
        return IFeedback(o1=o)

    def dist(self):
        if self.o is None:
            return self.a0model.dist()
        return self.amodel.dist(self.o)

    def pr(self, a):
        if self.o is None:
            return self.a0model.pr(a)
        return self.amodel.pr(self.o, a)

    def sample(self):
        if self.o is None:
            return self.a0model.sample()
        return self.amodel.sample(self.o)

    def plot(self, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.reactiveplot(self, nepisodes)
        self.idx = 0

    def plot_update(self):
        a0dist = self.a0model.probs()
        a0dist /= a0dist.sum(axis=-1, keepdims=True)

        adist = self.amodel.probs()
        adist /= adist.sum(axis=-1, keepdims=True)

        self.q.put((self.idx, a0dist, adist))
        self.idx += 1

        if self.idx == self.neps:
            self.q.put(None)

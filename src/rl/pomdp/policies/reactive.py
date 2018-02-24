from .policy import Policy

import pytk.factory.model as fmodel

from collections import namedtuple


IContext = namedtuple('IContext', 'o')
IFeedback = namedtuple('IFeedback', 'o1')

# TODO reset!!!!! how?! does one receive an observation to begin with?
import numpy as np


class Reactive(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.amodel = fmodel.Softmax(env.afactory, cond=(env.ofactory,))
        # TODO look at pgradient;  this won't work for some reason
        # self.params = self.amodel.params

    @property
    def params(self):
        return self.amodel.params

    @params.setter
    def params(self, value):
        self.amodel.params = value

    def reset(self):
        self.amodel.reset()

    def restart(self):
        # TODO how to select first obs state?
        # I think the environment gives an observation to you right away..?
        self.o = self.env.ofactory.item(0)

    @property
    def context(self):
        return IContext(self.o)

    def feedback(self, feedback):
        return self.feedback_o(feedback.o)

    def feedback_o(self, o):
        self.o = o
        return IFeedback(o1=o)

    def dist(self):
        return self.amodel.dist(self.o)

    def pr(self, a):
        return self.amodel.pr(self.o, a)

    def sample(self):
        return self.amodel.sample(self.o)

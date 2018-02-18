from .softmax import Softmax
import pytk.factory.model as fmodel

from pytk.util import argmax

import numpy as np


# TODO better inheritance stuff!! probably best to apply construction instead of inheritance

class PSoftmax(fmodel.Softmax):
    def __init__(self, env):
        super().__init__(env.afactory, cond=(env.sfactory,))
        self.env = env

    def amax(self, s, **kwargs):
        ddist = dict(self.dist(s))
        return argmax(
            lambda a:  ddist.get(a, 0.),
            self.env.actions,
            **kwargs
        )

# from .softmax import Softmax
import rl.misc.models as models

from pytk.util import argmax

import numpy as np


# TODO better inheritance stuff!! probably best to apply construction instead of inheritance

class PSoftmax(models.Softmax):
    def __init__(self, env):
        super().__init__(env.aspace, cond=(env.sspace,))
        self.env = env

    def reset(self):
        pass

    def restart(self):
        pass

    def amax(self, s, **kwargs):
        ddist = dict(self.dist(s))
        return argmax(
            lambda a:  ddist.get(a, 0.),
            self.env.actions,
            **kwargs
        )

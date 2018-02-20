import rl.values as rlvalues

from pytk.util import argmax

import numpy as np
import numpy.random as rnd
import functools
import math


class Q(rlvalues.Value):
    def value(self, s, a):
        return self[s, a]

    def __getitem__(self, sak):
        return self.qmodel[sak]

    def __setitem__(self, sak, value):
        self.qmodel[sak] = value

    def optim_value(self, s):
        return max(self.value(s, a) for a in self.env.actions)

    def optim_actions(self, s):
        return argmax(
                functools.partial(self.value, s),
                self.env.actions,
                all_=True,
            )

    def optim_action(self, s):
        return rnd.choice(self.optim_actions(s))


class Counts(rlvalues.Counts):
    def reset(self):
        self.cmodel[:] = 0

    def count(self, s, a):
        return self[s, a]

    def __getitem__(self, sak):
        return self.cmodel[sak]

    def __setitem__(self, sak, value):
        self.cmodel[sak] = value


class Eligibility(rlvalues.Eligibility):
    def reset(self):
        self.emodel[:] = 0.

    def elig(self, s, a):
        return self[s, a]

    def __getitem__(self, sak):
        return self.emodel[sak]

    def __setitem__(self, sak, value):
        self.emodel[sak] = value

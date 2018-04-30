from __future__ import division

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt


class Environment(object):
    def __init__(self):
        self.terminal = object()

    def actions(self, s):
        raise NotImplementedError

    def sample_s0(self):
        raise NotImplementedError

    def sample_s1(self, s0, a):
        raise NotImplementedError

    def sample_r(self, s0, a, s1):
        raise NotImplementedError


class StateSnake(State):
    def __init__(self, bodymask, foodmask):
        self.bodymask = bodymask
        self.foodmask = foodmask

    def __hash__(self):
        return hash((
            tuple(self.bodymask.ravel()),
            tuple(self.foodmask.ravel())
        ))

    @property
    def headpos(self):
        return np.argwhere(self.bodymask).ravel()


class SnakeEnv(Environment):
    """ Snake environment. """
    def __init__(self, shape):
        super(SnakeEnv, self).__init__()
        self.shape = shape
        self.size = np.prod(shape)

        self.scale_r = nfaces
        self.reroll_r = reroll_r

        bodymask = np.zeros(shape)
        bodymask[0, 0] = 1
        foodmask = np.zeros(shape)
        self.state_starting = SnakeState(bodymask, foodmask)

    def actions(self, s):
        return [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
        ]

    def sample_s0(self):
        return self.state_starting
    
    def sample_s1(self, s0, a):
        if a not in self.actions(s0):
            raise Exception

        s1headpos = s0.headpos + a
        if any(s1headpos < 0) or any(s1headpos >= self.shape):
            return s0

        s1foodmask = s0.foodmask.copy()
        s1bodymask = s0.bodymask.copy()

        if s0.foodmask[tuple(s1headpos)]:


        return self.terminal if a == 'stand' else DiceState(self.roll(), s0.nrolls + 1)

    def sample_r(self, s0, a, s1):
        if a not in self.actions(s0):
            raise Exception

        return float(s0.npips if a == 'stand' else self.reroll_r)

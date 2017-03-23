import itertools as itt

from pytk.decorators import memoizemethod

import numpy.random as rnd


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


class State(object):
    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self == other

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


class EnvironmentException(Exception):
    pass


class Environment(object):
    def __init__(self):
        self.terminal = object()

        self.__states_start = None
        self.__states_nostart = None

    @property
    def states_start(self):
        if self.__states_start is None:
            raise EnvironmentException('The starting states of this environment are not iterable.')
        return self.__states_start

    @states_start.setter
    def states_start(self, value):
        self.__states_start = value

    @property
    def states_nostart(self):
        if self.__states_nostart is None:
            raise EnvironmentException('The non-starting states of this environment are not iterable.')
        return self.__states_nostart

    @states_nostart.setter
    def states_nostart(self, value):
        self.__states_nostart = value

    @property
    def states(self):
        for s in itt.chain(self.states_start, self.states_nostart):
            yield s

    def actions(self, s):
        raise NotImplementedError

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


class ModelException(Exception):
    pass


class Model(object):
    """ Contains environment dynamics """
    def __init__(self, env):
        self.env = env
    
    def P(self, a, s0, s1):
        raise NotImplementedError

    def R(self, a, s0, s1):
        raise NotImplementedError

    def PR(self, a, s0, s1):
        return self.P(a, s0, s1), self.R(a, s0, s1)

    def PR_iter(self, s0, a):
        for s1 in self.env.states():
            p, r = self.PR(a, s0, s1)
            if p:
                yield p, s1, r

    def sample_s0(self):
        i = rnd.choice(self.env.nstates(begin=True, middle=False, terminal=False))
        return self.env.states(begin=True, middle=False, terminal=False)[i]

    def sample_s1(self, a, s0):
        ps, s1s = [], []
        for p, s1, _ in self.PR_iter(a, s0):
            ps.append(p)
            s1s.append(s1)

        i = rnd.choice(len(ps), p=ps)
        return s1s[i]
    
    def sample_r(self, a, s0, s1):
        raise NotImplementedError


class Model(object):
    def __init__(self, env):
        self.env = env

    def sample_s0(self):
        si = rnd.choice(len(self.env.states_start))
        return self.env.states_start[si]

    def sample_s1(self, s0, a):
        raise NotImplementedError

    def sample_r(self, s0, a, s1):
        raise NotImplementedError

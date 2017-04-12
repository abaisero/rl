import itertools as itt

from pytk.decorators import memoizemethod

import numpy.random as rnd


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###




### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

# self.terminal = object()
terminal = type(
    'Terminal',
    (object,),
    dict(__str__=lambda self: 'Terminal'),
)()



class EnvironmentException(Exception):
    pass


class Environment(object):
    def __init__(self):
        self.__states_start = None
        self.__states_nostart = None
        self.__actions_all = None

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
        return self.actions_all
    
    @property
    def actions_all(self):
        if self.__actions_all is None:
            raise EnvironmentException('The action list of this environment is not instantiated.  Maybe your environment doesn\'t impement the .actions(s) method?')
        return self.__actions_all

    @actions_all.setter
    def actions_all(self, value):
        self.__actions_all = value


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

    def sample_rs1(self, s0, a):
        s1 = self.sample_s1(s0, a)
        r = self.sample_r(s0, a, s1)
        return r, s1

from __future__ import division

import numpy.random as rnd 


class egreedy(object):
    def __init__(self, env, e=0):
        self.env = env
        self.e = e

        for s in env.states():
            print s, env.actions(s)
        self.pi = {s: env.actions(s)[0] for s in env.states()}

    def __getitem__(self, s):
        return self.pi[s]

    def __setitem__(self, s, a):
        self.pi[s] = a

    def sample(self, s):
        if self.e < rnd.random():
            return self[s]
        i = rnd.choice(self.env.nactions(s))
        return self.env.actions(s)[i]

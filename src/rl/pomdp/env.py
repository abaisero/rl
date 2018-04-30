from collections import namedtuple
from types import SimpleNamespace
import copy

import indextools
from rl_parsers.pomdp import parse as parse_pomdp

import rl.core as core
import rl.data as data

import numpy as np


Model = namedtuple('Model', 's0model, s1model, omodel, rewards')


class Environment:
    def __init__(self, name, sspace, aspace, ospace, model=None, gamma=None):
        self.name = name
        self.sspace = sspace
        self.aspace = aspace
        self.ospace = ospace
        self.model = model
        self.gamma = gamma

    @property
    def states(self):
        return self.sspace.elems

    @property
    def nstates(self):
        return self.sspace.nelems

    @property
    def actions(self):
        return self.aspace.elems

    @property
    def nactions(self):
        return self.aspace.nelems

    @property
    def obs(self):
        return self.ospace.elems

    @property
    def nobs(self):
        return self.ospace.nelems

    def new_context(self, gtype=None):
        s0, = self.model.s0model.sample()
        econtext = SimpleNamespace(t=0, s=s0, gtype=gtype)

        if gtype == 'longterm':
            econtext.g = 0.
        elif gtype == 'discounted':
            econtext.g = 0.
            econtext.discount = 1.

        return econtext

    def step(self, econtext, a, *, inline=True):
        t = econtext.t
        s = econtext.s

        t1 = t + 1
        s1, = self.model.s1model.sample(s, a)
        o, = self.model.omodel.sample(s, a, s1)
        r = self.model.rewards[s, a, s1]

        feedback = SimpleNamespace(r=r, o=o)
        if not inline:
            econtext = copy.copy(econtext)
        econtext.t = t1
        econtext.s = s1

        if econtext.gtype == 'longterm':
            econtext.g += (r - econtext.g) / t1
        elif econtext.gtype == 'discounted':
            econtext.g += r * econtext.discount
            econtext.discount *= self.gamma

        return feedback, econtext

    def episode(self, policy, nsteps, gtype=None):
        econtext = self.new_context(gtype)
        pcontext = policy.new_context()
        while econtext.t < nsteps:
            a = policy.sample_a(pcontext)
            feedback, _ = self.step(econtext, a)
            policy.step(pcontext, feedback)
        return econtext.g

    @staticmethod
    def from_fname(fname):
        with data.open_resource(fname, 'pomdp') as f:
            dotpomdp = parse_pomdp(f.read())

        if dotpomdp.values == 'cost':
            raise ValueError('I do not know how to handle `cost` values.')

        # TODO I think this should not be mean but something else..
        if np.any(dotpomdp.R.mean(axis=-1, keepdims=True) != dotpomdp.R):
            raise ValueError('I cannot handle rewards which depend on '
                             'observations.')

        name = fname
        sspace = indextools.DomainSpace(dotpomdp.states)
        aspace = indextools.DomainSpace(dotpomdp.actions)
        ospace = indextools.DomainSpace(dotpomdp.observations)
        gamma = dotpomdp.discount
        env = Environment(name, sspace, aspace, ospace, gamma=gamma)

        if dotpomdp.start is None:
            start = np.ones(sspace.nelems) / sspace.nelems
        else:
            start = dotpomdp.start
        T = np.swapaxes(dotpomdp.T, 0, 1)
        O = np.stack([dotpomdp.O] * sspace.nelems)
        R = np.einsum('jik', dotpomdp.R.mean(axis=-1))

        s0model = core.SpaceDistribution([], [sspace], start)
        s1model = core.SpaceDistribution([sspace, aspace], [sspace], T)
        omodel = core.SpaceDistribution([sspace, aspace, sspace], [ospace], O)
        rewards = R

        env.model = Model(s0model, s1model, omodel, rewards)
        return env

    def __repr__(self):
        return (f'POMDP({self.name}, |S|={self.nstates}, |A|={self.nactions}, '
                f'|O|={self.nobs})')

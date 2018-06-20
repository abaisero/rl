import logging

from collections import namedtuple

import indextools
from rl_parsers.pomdp import parse as parse_pomdp

import rl.data as data

import numpy as np

import torch
from torch.distributions import Categorical


Model = namedtuple('Model', 's0model, s1model, omodel, rewards')


class Environment:
    logger = logging.getLogger('rl.pomdp.Environment')

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

    def new(self, shape=(), *, device=torch.device('cpu')):
        return self.model.s0model.sample(shape).to(device)

    def step(self, s, a):
        device = s.device
        s = s.cpu()
        a = a.cpu()

        # NOTE following was buggy;  not truly independent samples
        # s1 = self.model.s1model.sample()[s, a]
        # o = self.model.omodel.sample()[s, a, s1]
        # r = self.model.rewards[s, a, s1]

        shape = s.shape
        size = s.nelement()
        idx = torch.arange(size).reshape(shape).long()

        s1 = self.model.s1model.sample((size,))[idx, s, a]
        o = self.model.omodel.sample((size,))[idx, s, a, s1]
        r = self.model.rewards[s, a, s1]

        # self.logger.debug(f'step():  {s} {a} -> {s1} {o} {r}')
        # return r, o, s1
        return r.to(device), o.to(device), s1.to(device)

    @staticmethod
    def from_fname(fname):
        with data.resource_open(fname, 'pomdp') as f:
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

        s0model = Categorical(probs=torch.tensor(start))
        s1model = Categorical(probs=torch.tensor(T))
        omodel = Categorical(probs=torch.tensor(O))
        rewards = torch.tensor(R)

        # import ipdb
        # ipdb.set_trace()

        env.model = Model(s0model, s1model, omodel, rewards)
        return env

    def __repr__(self):
        return (f'POMDP({self.name}, |S|={self.nstates}, |A|={self.nactions}, '
                f'|O|={self.nobs})')

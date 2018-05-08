import copy
import types

import indextools
from rl_parsers.fsc import parse as parse_fsc

import rl.data as data

import numpy as np
import numpy.random as rnd


class FSC_File:
    @staticmethod
    def from_namespace(env, namespace):
        return FSC_File.from_fname(env, namespace.fsc)

    @staticmethod
    def from_fname(env, fname):
        with data.open_resource(fname, 'fsc') as f:
            dotfsc = parse_fsc(f.read())

        # TODO stadt..
        nspace = indextools.DomainSpace(dotfsc.nodes)
        n0 = dotfsc.start.argmax()
        A = dotfsc.A
        T = np.swapaxes(dotfsc.T, 0, 1)
        return FSC_File(env, nspace, n0, A, T)

    def __init__(self, env, nspace, n0, A, T):
        self.aspace = env.aspace
        self.ospace = env.ospace
        self.nspace = nspace
        self.n0 = n0
        self.models = A, T

    @property
    def nodes(self):
        return self.nspace.elems

    @property
    def nnodes(self):
        return self.nspace.nelems

    @property
    def amodel(self):
        return self.models[0]

    @property
    def nmodel(self):
        return self.models[1]

    def new_context(self):
        return types.SimpleNamespace(n=self.n0)

    def step(self, pcontext, feedback, *, inline=False):
        n1 = self.sample_n1(pcontext, feedback)

        if not inline:
            pcontext = copy.copy(pcontext)
        pcontext.n = n1

        return pcontext

    def pr_a(self, pcontext):
        return self.amodel[pcontext.n]

    def sample_a(self, pcontext):
        probs = self.pr_a(pcontext)
        ai = rnd.multinomial(1, probs).argmax()
        return self.aspace.elem(ai)

    def pr_n(self, pcontext, feedback):
        return self.nmodel[pcontext.n, feedback.o]

    def sample_n1(self, pcontext, feedback):
        probs = self.pr_n(pcontext, feedback)
        ni = rnd.multinomial(1, probs).argmax()
        return self.nspace.elem(ni)

    def __repr__(self):
        return f'FSC(|N|={self.nnodes}, |A|={self.aspace.nelems})'

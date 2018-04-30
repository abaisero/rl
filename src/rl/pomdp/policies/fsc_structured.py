import copy
import types

import indextools
from rl_parsers.fss import parse as parse_fss

import rl.data as data
import rl.misc.models as models
import rl.graph as graph

import numpy as np


class FSC_Structured:
    @staticmethod
    def from_namespace(env, namespace):
        return FSC_Structured.from_fname(env, namespace.fss)

    @staticmethod
    def from_fname(env, fname):
        with data.open_resource(fname, 'fss') as f:
            dotfss = parse_fss(f.read())

        # TODO check that all actions are used
        # TODO check that structure fully connected
        # TODO check that the actions are the same

        if isinstance(dotfss.nodes[0], str):
            nodes = dotfss.nodes
        else:
            nodes = tuple(f'node_{n}' for n in dotfss.nodes)
        nspace = indextools.DomainSpace(nodes)

        return FSC_Structured(env, nspace, dotfss.start, dotfss.A, dotfss.N)

    def __init__(self, env, nspace, n0, amask, nmask):
        if not amask.ndim == 2:
            raise ValueError(
                f'Action mask has {amask.ndim} dimensions;  should have 2.')
        if not nmask.ndim == 2:
            raise ValueError(
                f'Node mask has {nmask.ndim} dimensions;  should have 3.')

        # TODO finish this!
        # if not amask.shape[0] == env.nactions:
        #     raise ValueError(f'Action mask shape {amask.shape} is wrong.')
        # if not amask.shape[1] == nmask.shape[0] == nmask.shape[1]:
        #     raise ValueError(f'Action mask shape {amask.shape} and/or node '
        #     'mask shape {nmask.shape} is wrong.')
        # if not n0mask.shape[0] == nmask.shape[0]

        self.aspace = env.aspace
        self.ospace = env.ospace
        self.nspace = nspace
        # TODO specific starting element???
        self.n0 = nspace.elem(n0)

        nmask = np.tile(
            nmask.reshape((self.nspace.nelems, 1, self.nspace.nelems)),
            (1, self.ospace.nelems, 1)
        )

        self.amask = amask
        self.nmask = nmask

        nn, na, no = self.nspace.nelems, self.aspace.nelems, self.ospace.nelems
        self.models = (
            models.Softmax([nn], [na], mask=amask),
            models.Softmax([nn, no], [nn], mask=nmask),
        )

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

    def new_params(self):
        params = np.empty(2, dtype=object)
        params[0] = self.models[0].new_params()
        params[1] = self.models[1].new_params()
        return params

    def new_context(self, params):
        return types.SimpleNamespace(n=self.n0)

    def step(self, params, pcontext, feedback, *, inline=True):
        n1 = self.sample_n(params, pcontext, feedback)

        if not inline:
            pcontext = copy.copy(pcontext)
        pcontext.n = n1

        return pcontext

    def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
        n, o, n1 = pcontext.n, feedback.o, pcontext1.n
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.models[0].dlogprobs(params[0], (n, a))
        dlogprobs[1] = self.models[1].dlogprobs(params[1], (n, o, n1))
        return dlogprobs

    def pr_a(self, params, pcontext):
        return self.amodel.pr(params[0], (pcontext.n,))

    def sample_a(self, params, pcontext):
        ai, = self.amodel.sample(params[0], (pcontext.n,))
        return self.aspace.elem(ai)

    def pr_n(self, params, pcontext, feedback):
        return self.nmodel.pr(params[1], (pcontext.n, feedback.o))

    def sample_n(self, params, pcontext, feedback):
        ni, = self.nmodel.sample(params[1], (pcontext.n, feedback.o))
        return self.nspace.elem(ni)

    def new_plot(self, nepisodes):
        return graph.FSC_Structured_Plotter(self, nepisodes)

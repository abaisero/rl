import copy
import types

import indextools
import rl.misc.models as models
import rl.graph as graph

import numpy as np


class FSC:
    @staticmethod
    def from_namespace(env, namespace):
        nodes = [f'node_{i}' for i in range(namespace.n)]
        nspace = indextools.DomainSpace(nodes)

        return FSC(env, nspace)

    def __init__(self, env, nspace):
        self.aspace = env.aspace
        self.ospace = env.ospace
        self.nspace = nspace

        nn, na, no = self.nspace.nelems, self.aspace.nelems, self.ospace.nelems

        self.models = (
            models.Softmax([nn], [na]),
            models.Softmax([nn, no], [nn]),
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

    def process_params(self, params, *, inline=False):
        if not inline:
            params = params.copy()

        self.models[0].process_params(params[0], inline=True)
        self.models[1].process_params(params[1], inline=True)
        return params

    def new_context(self, params):
        return types.SimpleNamespace(n=self.nspace.elem(0))

    def step(self, params, pcontext, feedback, *, inline=False):
        n1 = self.sample_n1(params, pcontext, feedback)

        if not inline:
            pcontext = copy.copy(pcontext)
        pcontext.n = n1

        return pcontext

    def logprobs(self, params, pcontext, a, feedback, pcontext1):
        n, o, n1 = pcontext.n, feedback.o, pcontext1.n
        logprobs = np.empty(2, dtype=object)
        logprobs[0] = self.models[0].logprobs(params[0], (n, a))
        logprobs[1] = self.models[1].logprobs(params[1], (n, o, n1))
        return logprobs

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

    def sample_n1(self, params, pcontext, feedback):
        ni, = self.nmodel.sample(params[1], (pcontext.n, feedback.o))
        return self.nspace.elem(ni)

    def new_plot(self, nepisodes):
        return graph.FSC_Plotter(self, nepisodes)

    def __repr__(self):
        return f'FSC(|N|={self.nnodes}, |A|={self.aspace.nelems})'

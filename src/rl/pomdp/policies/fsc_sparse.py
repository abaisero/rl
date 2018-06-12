import copy
import types
import itertools as itt

import indextools
import rl.misc.models as models
import rl.graph as graph

import numpy as np
import numpy.linalg as la
import numpy.random as rnd


class FSC_Sparse:
    @staticmethod
    def from_namespace(env, namespace):
        nodes = [f'node_{i}' for i in range(namespace.n)]
        nspace = indextools.DomainSpace(nodes)

        return FSC_Sparse(env, nspace, namespace.k)

    def __init__(self, env, nspace, K):
        self.aspace = env.aspace
        self.ospace = env.ospace
        self.nspace = nspace
        self.K = K

        combs = list(itt.combinations(range(env.nobs), 2))
        test_mask = np.zeros((env.nobs, len(combs)))
        for i, comb in enumerate(combs):
            test_mask[comb, i] = 1, -1

        for nfails in itt.count():
            if nfails == 100:
                raise ValueError(f'Could not initialize {self}')

            nmask = np.array([
                [rnd.permutation(self.nnodes) for _ in range(env.nobs)]
                for _ in range(self.nnodes)]) < K

            # check that graph is not disjoint
            _nn = nmask.sum(axis=1)
            test = la.multi_dot([_nn] * self.nnodes)
            if np.any(test == 0):
                continue

            # check that each observation gives a different transition mask
            test = np.einsum('hyg,yn->hng', nmask, test_mask)
            if np.all(test == 0, axis=0).any():
                continue

            break

        self.nmask = nmask

        nn, na, no = self.nspace.nelems, self.aspace.nelems, self.ospace.nelems
        self.models = (
            models.Softmax([nn], [na]),
            models.Softmax([nn, no], [nn], mask=nmask),
        )

    @property
    def nodes(self):
        return self.nspace.nelems

    @property
    def nnodes(self):
        return self.nspace.nelems

    @property
    def nmodels(self):
        return len(self.models)

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
        n1 = self.sample_n(params, pcontext, feedback)

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

    def sample_n(self, params, pcontext, feedback):
        ni, = self.nmodel.sample(params[1], (pcontext.n, feedback.o))
        return self.nspace.elem(ni)

    def new_plot(self, nepisodes):
        return graph.FSC_Sparse_Plotter(self, nepisodes)

    def __repr__(self):
        return f'FSC_Sparse(N={self.nnodes}, K={self.K})'

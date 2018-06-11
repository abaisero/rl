import copy
import types

import rl.misc.models as models

import numpy as np


class FSC_Reactive:
    @staticmethod
    def from_namespace(env, namespace):
        return FSC_Reactive(env, namespace.k)

    def __init__(self, env, k):
        self.k = k
        self.aspace = env.aspace
        self.ospace = env.ospace

        self.models = tuple(
            models.Softmax(i * [env.ospace.nelems], [self.aspace.nelems])
            for i in range(k + 1)
        )

    def new_params(self):
        params = np.empty(self.k + 1, dtype=object)
        for i in range(self.k + 1):
            params[i] = self.models[i].new_params()
        return params

    def process_params(self, params, *, inline=False):
        if not inline:
            params = params.copy()

        for i in range(self.k + 1):
            self.models[i].process_params(params[i], inline=True)
        return params

    def new_context(self, params):
        return types.SimpleNamespace(hist=())

    def step(self, params, pcontext, feedback, *, inline=False):
        if not inline:
            pcontext = copy.copy(pcontext)

        if len(pcontext.hist) == self.k:
            pcontext.hist = pcontext.hist[1:]
        pcontext.hist += feedback.o,

        return pcontext

    def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
        hlen = len(pcontext.hist)

        dlogprobs = np.full(self.k + 1, 0., dtype=object)
        for i in range(self.k + 1):
            dlogprobs[i] = np.zeros(1)
        dlogprobs[hlen] = self.models[hlen].dlogprobs(params[hlen],
                                                      pcontext.hist + (a,))
        return dlogprobs

    def pr_a(self, params, pcontext):
        hlen = len(pcontext.hist)
        return self.models[hlen].pr(params[hlen], pcontext.hist)

    def sample_a(self, params, pcontext):
        hlen = len(pcontext.hist)
        ai, = self.models[hlen].sample(params[hlen], pcontext.hist)
        return self.aspace.elem(ai)

    # TODO plot stuff
    # def new_plot(self, nepisodes):
    #     return graph.Reactive_Plotter(self, nepisodes)

    def __repr__(self):
        return f'FSC_Reactive(K={self.k})'

import rl.graph as graph

import types

import rl.misc.models as models

import numpy as np


class CF:
    @staticmethod
    def from_namespace(env, namespace):
        return CF(env)

    def __init__(self, env):
        self.aspace = env.aspace
        self.models = models.Softmax([], [self.aspace.nelems]),

    @property
    def amodel(self):
        return self.models[0]

    def new_params(self):
        params = np.empty(1, dtype=object)
        params[0] = self.models[0].new_params()
        return params

    def new_context(self, params):
        return types.SimpleNamespace()

    def step(self, params, pcontext, feedback, *, inline=False):
        return pcontext

    def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
        dlogprobs = np.empty(1, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(params[0], (a,))
        return dlogprobs

    def pr_a(self, params, pcontext):
        return self.amodel.pr(params[0], ())

    def sample_a(self, params, pcontext):
        ai, = self.amodel.sample(params[0], ())
        return self.aspace.elem(ai)

    def new_plot(self, nepisodes):
        return graph.CF_Plotter(self, nepisodes)

    def __repr__(self):
        return f'FSC(|A|={self.aspace.nelems})'

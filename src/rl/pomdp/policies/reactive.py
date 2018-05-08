import copy
import types

import rl.misc.models as models
import rl.graph as graph

import numpy as np


class Reactive:
    @staticmethod
    def from_namespace(env, namespace):
        return Reactive(env)

    def __init__(self, env):
        self.aspace = env.aspace
        self.ospace = env.ospace
        self.models = (
            models.Softmax([], [self.aspace.nelems]),
            models.Softmax([self.ospace.nelems], [self.aspace.nelems]),
        )

    @property
    def a0model(self):
        return self.models[0]

    @property
    def amodel(self):
        return self.models[1]

    def new_params(self):
        params = np.empty(2, dtype=object)
        params[0] = self.models[0].new_params()
        params[1] = self.models[1].new_params()
        return params

    def new_context(self, params):
        return types.SimpleNamespace(o=None)

    def step(self, params, pcontext, feedback, *, inline=False):
        if not inline:
            pcontext = copy.copy(pcontext)
        pcontext.o = feedback.o

        return pcontext

    def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
        o = pcontext.o

        dlogprobs = np.full(2, 0., dtype=object)
        if o is None:
            dlogprobs[0] = self.a0model.dlogprobs(params[0], (a,))
        else:
            dlogprobs[1] = self.amodel.dlogprobs(params[1], (o, a))
        return dlogprobs

    def pr_a(self, params, pcontext):
        if pcontext.o is None:
            return self.a0model.pr(params[0], ())
        else:
            return self.amodel.pr(params[1], (pcontext.o,))

    def sample_a(self, params, pcontext):
        if pcontext.o is None:
            ai, = self.a0model.sample(params[0], ())
        else:
            ai, = self.amodel.sample(params[1], (pcontext.o,))
        return self.aspace.elem(ai)

    def new_plot(self, nepisodes):
        return graph.Reactive_Plotter(self, nepisodes)

    def __repr__(self):
        return f'Reactive(|O|={self.ospace.nelems})'

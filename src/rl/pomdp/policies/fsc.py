# import copy
# import types

# import indextools
# import rl.misc.models as models
# import rl.graph as graph

# import numpy as np


# class FSC:
#     @staticmethod
#     def from_namespace(env, namespace):
#         nodes = [f'node_{i}' for i in range(namespace.n)]
#         nspace = indextools.DomainSpace(nodes)

#         return FSC(env, nspace)

#     def __init__(self, env, nspace):
#         self.aspace = env.aspace
#         self.ospace = env.ospace
#         self.nspace = nspace

#         nn, na, no = self.nspace.nelems, self.aspace.nelems, self.ospace.nelems

#         self.models = (
#             models.Softmax([nn], [na]),
#             models.Softmax([nn, no], [nn]),
#         )

#     @property
#     def nodes(self):
#         return self.nspace.elems

#     @property
#     def nnodes(self):
#         return self.nspace.nelems

#     @property
#     def amodel(self):
#         return self.models[0]

#     @property
#     def nmodel(self):
#         return self.models[1]

#     def new_params(self):
#         params = np.empty(2, dtype=object)
#         params[0] = self.models[0].new_params()
#         params[1] = self.models[1].new_params()
#         return params

#     def process_params(self, params, *, inline=False):
#         if not inline:
#             params = params.copy()

#         self.models[0].process_params(params[0], inline=True)
#         self.models[1].process_params(params[1], inline=True)
#         return params

#     def new_context(self, params):
#         return types.SimpleNamespace(n=self.nspace.elem(0))

#     def step(self, params, pcontext, feedback, *, inline=False):
#         n1 = self.sample_n1(params, pcontext, feedback)

#         if not inline:
#             pcontext = copy.copy(pcontext)
#         pcontext.n = n1

#         return pcontext

#     def logprobs(self, params, pcontext, a, feedback, pcontext1):
#         n, o, n1 = pcontext.n, feedback.o, pcontext1.n
#         logprobs = np.empty(2, dtype=object)
#         logprobs[0] = self.models[0].logprobs(params[0], (n, a))
#         logprobs[1] = self.models[1].logprobs(params[1], (n, o, n1))
#         return logprobs

#     def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
#         n, o, n1 = pcontext.n, feedback.o, pcontext1.n
#         dlogprobs = np.empty(2, dtype=object)
#         dlogprobs[0] = self.models[0].dlogprobs(params[0], (n, a))
#         dlogprobs[1] = self.models[1].dlogprobs(params[1], (n, o, n1))
#         return dlogprobs

#     def pr_a(self, params, pcontext):
#         return self.amodel.pr(params[0], (pcontext.n,))

#     def sample_a(self, params, pcontext):
#         ai, = self.amodel.sample(params[0], (pcontext.n,))
#         return self.aspace.elem(ai)

#     def pr_n(self, params, pcontext, feedback):
#         return self.nmodel.pr(params[1], (pcontext.n, feedback.o))

#     def sample_n1(self, params, pcontext, feedback):
#         ni, = self.nmodel.sample(params[1], (pcontext.n, feedback.o))
#         return self.nspace.elem(ni)

#     def new_plot(self, nepisodes):
#         return graph.FSC_Plotter(self, nepisodes)

#     def __repr__(self):
#         return f'FSC(|N|={self.nnodes}, |A|={self.aspace.nelems})'


from .astrat import AStrategy
from .ostrat import OStrategy
from .nvalue import Value

import torch
import torch.nn as nn
from torch.distributions import Categorical


class FSC:
    def __init__(self, env, nnodes, gain=1., critic=False, device=torch.device('cpu')):
        super().__init__()
        self.env = env
        # self.nspace = indextools.RangeSpace(nnodes)

        # TODO how to handle shared stuff..?
        # self._nshare = Identity()
        self.astrat = AStrategy(nnodes, env.nactions, gain=gain)
        self.ostrat = OStrategy(nnodes, env.nobs, gain=gain, device=device)

        self.modules = nn.ModuleList()
        # self.modules.add_module('nshare', self._nshare)
        self.modules.add_module('astrat', self.astrat)
        self.modules.add_module('ostrat', self.ostrat)

        self.critic = None
        if critic:
            self.critic = Value(nnodes)

            self.modules.add_module('critic', self.critic)

    def parameters(self, config=None):
        def rgfilter(parameters):
            return filter(lambda p: p.requires_grad, parameters)

        if config is None:
            return rgfilter(self.modules.parameters())

        parameters = []

        pdict = {'params': rgfilter(self.astrat.parameters())}
        if config.lra is not None:
            pdict['lr'] = config.lra
        parameters.append(pdict)

        pdict = {'params': rgfilter(self.ostrat.parameters())}
        if config.lrb is not None:
            pdict['lr'] = config.lrb
        parameters.append(pdict)

        if self.critic:
            pdict = {'params': rgfilter(self.critic.parameters())}
            if config.lrc is not None:
                pdict['lr'] = config.lrc
            parameters.append(pdict)

        return parameters

    def new(self, shape=(), *, device=torch.device('cpu')):
        n = torch.zeros(shape).to(device, torch.long)
        nnll = torch.zeros(shape).to(device)
        return n, nnll

    def act(self, n):
        probs = self.astrat(n)
        # TODO use.... logits instead?
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(n).squeeze(-1)

        return sample, nll

    def step(self, n, o):
        probs = self.ostrat(n, o)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(sample).squeeze(-1)

        return sample, nll


class FSC:
    def __init__(self, astrat, ostrat, *, istrat=None, critic=None):
        super().__init__()

        self.ml = nn.ModuleList()
        self.ml.add_module('astrat', astrat)
        self.ml.add_module('ostrat', ostrat)
        self.ml.add_module('critic', critic)

    def parameters(self, config=None):
        def rgfilter(parameters):
            return filter(lambda p: p.requires_grad, parameters)

        if config is None:
            return rgfilter(self.ml.parameters())

        parameters = []

        pdict = {'params': rgfilter(self.ml.astrat.parameters())}
        if config.lra is not None:
            pdict['lr'] = config.lra
        parameters.append(pdict)

        pdict = {'params': rgfilter(self.ml.ostrat.parameters())}
        if config.lrb is not None:
            pdict['lr'] = config.lrb
        parameters.append(pdict)

        if self.ml.critic is not None:
            pdict = {'params': rgfilter(self.ml.critic.parameters())}
            if config.lrc is not None:
                pdict['lr'] = config.lrc
            parameters.append(pdict)

        return parameters

    def new(self, shape=(), *, device=torch.device('cpu')):
        n = torch.zeros(shape).to(device, torch.long)
        nnll = torch.zeros(shape).to(device)
        return n, nnll

    def act(self, n):
        a, anll = self.ml.astrat.sample(n)

        if self.ml.critic is not None:
            return a, anll, self.value(n)

        return a, anll

    def anll(self, n, a):
        return self.ml.astrat.nll(n, a)

    def value(self, n):
        return self.ml.critic(n).squeeze(-1)

    def step(self, n, o):
        n1, nnll = self.ml.ostrat.sample(n, o)

        if self.ml.critic is not None:
            return n1, nnll, self.value(n1)

        return n1, nnll

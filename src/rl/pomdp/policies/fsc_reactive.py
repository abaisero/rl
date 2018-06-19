# import copy
# import types

# import rl.misc.models as models

# import numpy as np


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


# NOTE this has been made rendundant
# class Reactive:
#     def __init__(self, env, gain=1., critic=False):
#         super().__init__()
#         self.env = env

#         self.nnodes = env.nobs + 1
#         self.astrat = AStrategy(self.nnodes, env.nactions, gain=gain)

#         self.modules = nn.ModuleList()
#         self.modules.add_module('astrat', self.astrat)

#         self.critic = None
#         if critic:
#             self.critic = Value(self.nnodes)
#             self.modules.add_module('critic', self.critic)

#     def parameters(self, config=None):
#         def rgfilter(parameters):
#             return filter(lambda p: p.requires_grad, parameters)

#         if config is None:
#             return rgfilter(self.modules.parameters())

#         parameters = []

#         pdict = {'params': rgfilter(self.astrat.parameters())}
#         if config.lra is not None:
#             pdict['lr'] = config.lra
#         parameters.append(pdict)

#         pdict = {'params': rgfilter(self.ostrat.parameters())}
#         if config.lrb is not None:
#             pdict['lr'] = config.lrb
#         parameters.append(pdict)

#         if self.critic:
#             pdict = {'params': rgfilter(self.critic.parameters())}
#             if config.lrc is not None:
#                 pdict['lr'] = config.lrc
#             parameters.append(pdict)

#         return parameters

#     def new(self, shape=()):
#         return torch.full(shape, 0).long(), torch.zeros(shape)

#     def act(self, n):
#         probs = self.astrat(n)
#         dist = Categorical(probs)
#         sample = dist.sample()
#         nll = -dist.log_prob(sample)

#         if self.critic:
#             return sample, nll, self.critic(n)

#         return sample, nll

#     def step(self, n, o):
#         n1 = o + 1

#         if self.critic:
#             return n1, 0., self.critic(n1)

#         return n1, 0.


# # TODO actually this whole thing is just a combination of astrat and ostrat
# class FSC_Reactive:
#     def __init__(self, env, K, gain=1., critic=False):
#         super().__init__()
#         self.env = env
#         self.K = K

#         # TODO triple check!!!
#         self._no = env.nobs
#         self._mod = self._no ** (K - 1)
#         self._bases = torch.cat([
#             torch.zeros((1,), dtype=torch.long),
#             torch.full((K,), self._no).long().cumprod(0).cumsum(0) / self._no,
#         ])
#         self._bases_extra = torch.cat([
#             self._bases,
#             self._bases[-1].unsqueeze(0),
#         ])
#         self._decode_key = self._no ** torch.arange(K - 1, -1, -1).long()

#         self.nnodes = self._bases[-1] + self._no ** K
#         self.astrat = AStrategy(self.nnodes, env.nactions, gain=gain)

#         self.modules = nn.ModuleList()
#         self.modules.add_module('astrat', self.astrat)

#         self.critic = None
#         if critic:
#             self.critic = Value(self.nnodes)
#             self.modules.add_module('critic', self.critic)

#     def parameters(self, config=None):
#         def rgfilter(parameters):
#             return filter(lambda p: p.requires_grad, parameters)

#         if config is None:
#             return rgfilter(self.modules.parameters())

#         parameters = []

#         pdict = {'params': rgfilter(self.astrat.parameters())}
#         if config.lra is not None:
#             pdict['lr'] = config.lra
#         parameters.append(pdict)

#         if self.critic:
#             pdict = {'params': rgfilter(self.critic.parameters())}
#             if config.lrc is not None:
#                 pdict['lr'] = config.lrc
#             parameters.append(pdict)

#         return parameters

#     def new(self, shape=()):
#         return torch.full(shape, 0).long(), torch.zeros(shape)

#     def act(self, n):
#         probs = self.astrat(n)
#         dist = Categorical(probs)
#         sample = dist.sample()
#         nll = -dist.log_prob(sample)

#         if self.critic:
#             return sample, nll, self.critic(n).squeeze(-1)

#         return sample, nll

#     def step(self, n, o):
#         n1 = self._step(n, o)

#         if self.critic:
#             return n1, 0., self.critic(n1).squeeze(-1)

#         return n1, 0.

#     def _encode(self, os):
#         raise NotImplementedError

#     def _decode(self, n):
#         ibase = self._bases.le(n.unsqueeze(-1)).sum(1) - 1
#         base = self._bases[ibase]
#         os_full = self._decode_full(n - base)
#         cond = torch.stack([torch.arange(self.K).long()] * len(n))
#         cond.lt_((self.K-ibase).unsqueeze(-1))
#         return torch.where(cond.byte(), torch.tensor(-1).to(config.device), os_full.t())

#     def _decode_full(self, code):
#         # TODO I should transpose here, not outside
#         return code.div(self._decode_key.unsqueeze(-1)) % self._no

#     def _step(self, n, o):
#         # rule for k-order reactive internal dynamics
#         ibase = self._bases.le(n.unsqueeze(-1)).sum(1) - 1
#         base = self._bases[ibase]
#         base1 = self._bases_extra[ibase + 1]
#         return ((n - base) % self._mod) * self._no + base1 + o

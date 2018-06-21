import torch
import torch.nn as nn

from rl.utils import wtuple


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

    def new(self, shape=()):
        n = torch.zeros(shape).long()
        nnll = torch.zeros(shape)
        return wtuple(n, nnll)

    def act(self, n):
        a, anll = self.ml.astrat.sample(n)

        if self.ml.critic is not None:
            return wtuple(a, anll, self.value(n))

        return wtuple(a, anll)

    def anll(self, n, a):
        return self.ml.astrat.nll(n, a)

    def value(self, n):
        return self.ml.critic(n).squeeze(-1)

    def step(self, n, o):
        n1, nnll = self.ml.ostrat.sample(n, o)

        if self.ml.critic is not None:
            return wtuple(n1, nnll, self.value(n1))

        return wtuple(n1, nnll)

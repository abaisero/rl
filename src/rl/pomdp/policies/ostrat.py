import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .multi_embedding import MultiEmbedding_from_pretrained

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import itertools as itt


def make_mask(nnodes, nobs, K):
    # TODO change to pytorch tensor?
    combs = list(itt.combinations(range(nobs), 2))
    test_mask = np.zeros((nobs, len(combs)))
    for i, comb in enumerate(combs):
        test_mask[comb, i] = 1, -1
    # test_mask = torch.from_numpy(test_mask)

    for nfails in itt.count():
        if nfails == 100:
            raise ValueError('Could not make mask.')

        nmask = np.array([
            [rnd.permutation(nnodes) for _ in range(nobs)]
            for _ in range(nnodes)]) < K
        # nmask = torch.stack([
        #     torch.stack([torch.randperm(nnodes) for _ in range(nobs)])
        #     for _ in range(nnodes)]) < K

        # check that graph is not disjoint
        _nn = nmask.sum(axis=1)
        test = la.multi_dot([_nn] * nnodes)
        if np.any(test == 0):
            continue

        # # check that graph is not disjoint
        # _nn = nmask.sum(dim=1)
        # test = torch.eye(nnodes).long()
        # for _ in range(nnodes):
        #     test = test @ _nn
        # if (test == 0).any():
        #     continue

        # check that each observation gives a different transition mask
        test = np.einsum('hyg,yn->hng', nmask, test_mask)
        if np.all(test == 0, axis=0).any():
            continue

        break

    return torch.from_numpy(nmask.astype(np.uint8))


class OStrategy(nn.Module):
    def __init__(self, nnodes, nobs, gain=1.):
        super().__init__()
        embeddings = torch.eye(nnodes * nobs).reshape(nnodes, nobs, -1)
        self.embedding = MultiEmbedding_from_pretrained(embeddings=embeddings)
        self.linear = nn.Linear(nnodes * nobs, nnodes, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain)

    def forward(self, n, o):
        em = self.embedding(n, o)
        scores = self.linear(em)
        return F.log_softmax(scores, dim=-1)

    def sample(self, n, o):
        logits = self(n, o)
        dist = Categorical(logits=logits)
        sample = dist.sample()
        nll = -dist.log_prob(sample)
        return sample, nll


class OStrategy_Sparse(nn.Module):
    def __init__(self, nnodes, nobs, K, gain=1.):
        super().__init__()
        embeddings = torch.eye(nnodes * nobs).reshape(nnodes, nobs, -1)
        self.embedding = MultiEmbedding_from_pretrained(embeddings=embeddings)
        self.linear = nn.Linear(nnodes * nobs, nnodes, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain)

        membeddings = torch.where(
            make_mask(nnodes, nobs, K),
            torch.tensor(0.),
            torch.tensor(-float('inf')),
        )
        self.membedding = MultiEmbedding_from_pretrained(
            embeddings=membeddings)

    def forward(self, n, o):
        em = self.embedding(n, o)
        scores = self.linear(em)

        # TODO maybe initialization should also take the mask into acocunt
        mscores = self.membedding(n, o)
        return F.log_softmax(scores + mscores, dim=-1)

    def sample(self, n, o):
        logits = self(n, o)
        dist = Categorical(logits=logits)
        sample = dist.sample()
        nll = -dist.log_prob(sample)
        return sample, nll


class OStrategy_Reactive(nn.Module):
    def __init__(self, O, K):
        super().__init__()
        self.O = O
        self.K = K

        # TODO triple check!!!
        self.M = O ** (K - 1)
        self._bases = torch.cat([
            torch.zeros((1,), dtype=torch.long),
            torch.full((K,), O, dtype=torch.long).cumprod(0).cumsum(0) / O,
        ])
        self._bases_extra = torch.cat([
            self._bases,
            self._bases[-1].unsqueeze(0),
        ])
        self._decode_key = O ** torch.arange(K - 1, -1, -1).long()

        self.nnodes = self._bases[-1] + O ** K

    def sample(self, n, o):
        n1 = self._step(n, o)
        return n1, torch.tensor(0.).to(n1)

    def _encode(self, os):
        raise NotImplementedError

    def _decode(self, n):
        ibase = self._bases.to(n).le(n.unsqueeze(-1)).sum(1) - 1
        base = self._bases[ibase].to(n)
        os_full = self._decode_full(n - base)
        cond = torch.stack([torch.arange(self.K).long()] * len(n))
        cond.lt_((self.K - ibase).unsqueeze(-1))
        return torch.where(cond.byte(), torch.tensor(-1).to(n), os_full.t())

    def _decode_full(self, code):
        # TODO I should transpose here, not outside
        return code.div(self._decode_key.unsqueeze(-1)) % self.O

    def _step(self, n, o):
        # rule for k-order reactive internal dynamics
        ibase = self._bases.to(n).le(n.unsqueeze(-1)).sum(1) - 1
        base = self._bases[ibase].to(n)
        base1 = self._bases_extra[ibase + 1].to(n)
        return ((n - base) % self.M) * self.O + base1 + o

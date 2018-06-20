import argparse
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as tmp
from torch.utils.data import Dataset, DataLoader

import numpy as np

import rl.pomdp as pomdp
import rl.pomdp.policies as policies


parser = argparse.ArgumentParser(description='PyTorch FSC example')

parser.add_argument('--seed', type=int, default=None)

parser.add_argument('env', type=str, help='environment')
parser.add_argument('policy', type=str, help='policy')
parser.add_argument('algo', type=str,
                    choices=['vanilla', 'baseline', 'actorcritic', 'acl'],
                    default='vanilla')

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--samples', type=int, default=1)
parser.add_argument('--steps', type=int, default=100)

parser.add_argument('--noreplay', action='store_true')

parser.add_argument('--optim', type=str, choices=['sgd', 'adam', 'adamax'],
                    default='sgd')
parser.add_argument('--lr', type=float, default=1.)
parser.add_argument('--lra', type=float, default=None)
parser.add_argument('--lrb', type=float, default=None)
parser.add_argument('--lrc', type=float, default=None)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--clip', type=float, default='inf')

parser.add_argument('--gtype', type=str,
                    choices=['episodic', 'discounted', 'longterm'],
                    default='longterm')

parser.add_argument('--gain', type=float, default=1.)

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

config = parser.parse_args()

def make_r2g(n, gamma):
    r2g = gamma ** (-np.subtract.outer(range(n), range(n)))
    tril_idx = np.tril_indices(n, -1)
    r2g[tril_idx] = 0.
    return torch.from_numpy(r2g.astype(np.float32))


# class Identity(nn.Module):
#     def forward(Self, inputs):
#         return inputs


# class MultiEmbedding_from_pretrained(nn.Module):
#     def __init__(self, embeddings):
#         super().__init__()
#         idims = torch.tensor(embeddings.shape[:-1])
#         odim = embeddings.shape[-1]

#         self.midx = idims.cumprod(dim=0).unsqueeze(dim=0) / idims
#         embeddings_ = embeddings.reshape((-1, odim))
#         self.embedding = nn.Embedding.from_pretrained(embeddings_)

#     def forward(self, *codes):
#         code = torch.stack(codes, dim=-1)
#         # TODO change this to a standard batched product!  cuda does not allow integer stuff
#         # print('---')
#         # print(F.linear(code, self.midx).squeeze(dim=-1))
#         # print((code * self.midx).sum(1))

#         # import ipdb;  ipdb.set_trace()
#         # code = F.linear(code, self.midx).squeeze(dim=-1)
#         code = (code * self.midx).sum(1)
#         # print('code', code)
#         em = self.embedding(code)
#         # print('em', em)
#         return em


# class AStrategy(nn.Module):
#     def __init__(self, nnodes, nactions, gain=1.):
#         super().__init__()
#         embeddings = torch.eye(nnodes)
#         self.embedding = nn.Embedding.from_pretrained(embeddings)
#         self.linear = nn.Linear(nnodes, nactions, bias=False)
#         nn.init.xavier_normal_(self.linear.weight, gain)

#     def forward(self, n):
#         em = self.embedding(n)
#         scores = self.linear(em)
#         return F.softmax(scores, dim=-1)


# class OStrategy(nn.Module):
#     def __init__(self, nnodes, nobs, gain=1.):
#         super().__init__()
#         embeddings = torch.eye(nnodes * nobs).reshape(nnodes, nobs, -1)
#         self.embedding = MultiEmbedding_from_pretrained(embeddings=embeddings)
#         self.linear = nn.Linear(nnodes * nobs, nnodes, bias=False)
#         nn.init.xavier_normal_(self.linear.weight, gain)

#     def forward(self, n, o):
#         em = self.embedding(n, o)
#         scores = self.linear(em)
#         return F.softmax(scores, dim=-1)


# import numpy.linalg as la
# import numpy.random as rnd
# import itertools as itt


# def make_mask(nnodes, nobs, K):
#     # TODO change to pytorch tensor?
#     combs = list(itt.combinations(range(nobs), 2))
#     test_mask = np.zeros((nobs, len(combs)))
#     for i, comb in enumerate(combs):
#         test_mask[comb, i] = 1, -1
#     # test_mask = torch.from_numpy(test_mask)

#     for nfails in itt.count():
#         if nfails == 100:
#             raise ValueError('Could not make mask.')

#         nmask = np.array([
#             [rnd.permutation(nnodes) for _ in range(nobs)]
#             for _ in range(nnodes)]) < K
#         # nmask = torch.stack([
#         #     torch.stack([torch.randperm(nnodes) for _ in range(nobs)])
#         #     for _ in range(nnodes)]) < K

#         # check that graph is not disjoint
#         _nn = nmask.sum(axis=1)
#         test = la.multi_dot([_nn] * nnodes)
#         if np.any(test == 0):
#             continue

#         # # check that graph is not disjoint
#         # _nn = nmask.sum(dim=1)
#         # test = torch.eye(nnodes).long()
#         # for _ in range(nnodes):
#         #     test = test @ _nn
#         # if (test == 0).any():
#         #     continue

#         # check that each observation gives a different transition mask
#         test = np.einsum('hyg,yn->hng', nmask, test_mask)
#         if np.all(test == 0, axis=0).any():
#             continue

#         break

#     return torch.from_numpy(nmask.astype(np.uint8))


# class OStrategy_Masked(nn.Module):
#     def __init__(self, nnodes, nobs, K, gain=1.):
#         super().__init__()
#         embeddings = torch.eye(nnodes * nobs).reshape(nnodes, nobs, -1)
#         self.embedding = MultiEmbedding_from_pretrained(embeddings=embeddings)
#         self.linear = nn.Linear(nnodes * nobs, nnodes, bias=False)
#         nn.init.xavier_normal_(self.linear.weight, gain)

#         membeddings = torch.where(
#             make_mask(nnodes, nobs, K),
#             torch.tensor(0.),
#             torch.tensor(-float('inf'))
#         )
#         self.membedding = MultiEmbedding_from_pretrained(
#             embeddings=membeddings)

#     def forward(self, n, o):
#         em = self.embedding(n, o)
#         scores = self.linear(em)

#         # TODO maybe initialization should also take the mask into acocunt
#         mscores = self.membedding(n, o)
#         return F.softmax(scores + mscores, dim=-1)


# class Value(nn.Module):
#     def __init__(self, nnodes, gain=1.):
#         super().__init__()
#         embeddings = torch.eye(nnodes)
#         self.embedding = nn.Embedding.from_pretrained(embeddings)
#         self.affine = nn.Linear(nnodes, 1, bias=True)
#         nn.init.xavier_normal_(self.affine.weight, gain)
#         nn.init.constant_(self.affine.bias, 0)

#     def forward(self, n):
#         em = self.embedding(n)
#         return self.affine(em)


# class FSC:
#     def __init__(self, env, nnodes, gain=1., critic=False):
#         super().__init__()
#         self.env = env
#         # self.nspace = indextools.RangeSpace(nnodes)

#         # self._nshare = Identity()
#         self.astrat = AStrategy(nnodes, env.nactions, gain=gain)
#         self.ostrat = OStrategy(nnodes, env.nobs, gain=gain)

#         self.modules = nn.ModuleList()
#         # self.modules.add_module('nshare', self._nshare)
#         self.modules.add_module('astrat', self.astrat)
#         self.modules.add_module('ostrat', self.ostrat)

#         self.critic = None
#         if critic:
#             self.critic = Value(nnodes)
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
#             return sample, nll, self.critic(n).squeeze(-1)

#         return sample, nll

#     def anll(self, n, a):
#         probs = self.astrat(n)
#         dist = Categorical(probs)
#         return -dist.log_prob(a)

#     def step(self, n, o):
#         probs = self.ostrat(n, o)
#         dist = Categorical(probs)
#         sample = dist.sample()
#         nll = -dist.log_prob(sample)

#         if self.critic:
#             return sample, nll, self.critic(sample).squeeze(-1)

#         return sample, nll


# class FSC_Sparse:
#     def __init__(self, env, nnodes, K, gain=1., critic=False):
#         super().__init__()
#         self.env = env

#         self.astrat = AStrategy(nnodes, env.nactions, gain=gain)
#         self.ostrat = OStrategy_Masked(nnodes, env.nobs, K, gain=gain)

#         self.modules = nn.ModuleList()
#         self.modules.add_module('astrat', self.astrat)
#         self.modules.add_module('ostrat', self.ostrat)

#         self.critic = None
#         if critic:
#             self.critic = Value(nnodes)
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
#             return sample, nll, self.critic(n).squeeze(-1)

#         return sample, nll

#     def anll(self, n, a):
#         probs = self.astrat(n)
#         dist = Categorical(probs)
#         return -dist.log_prob(a)

#     def step(self, n, o):
#         probs = self.ostrat(n, o)
#         dist = Categorical(probs)
#         sample = dist.sample()
#         nll = -dist.log_prob(sample)

#         if self.critic:
#             return sample, nll, self.critic(sample).squeeze(-1)

#         return sample, nll


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

#     def anll(self, n, a):
#         probs = self.astrat(n)
#         dist = Categorical(probs)
#         return -dist.log_prob(a)

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
#         return torch.where(cond.byte(), torch.tensor(-1), os_full.t())

#     def _decode_full(self, code):
#         # TODO I should transpose here, not outside
#         return code.div(self._decode_key.unsqueeze(-1)) % self._no

#     def _step(self, n, o):
#         # rule for k-order reactive internal dynamics
#         ibase = self._bases.le(n.unsqueeze(-1)).sum(1) - 1
#         base = self._bases[ibase]
#         base1 = self._bases_extra[ibase + 1]
#         return ((n - base) % self._mod) * self._no + base1 + o


def evaluate(label, env, policy, samples, steps):
    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]

    s = env.new((samples,))
    n = policy.new((samples,))[0]

    rewards = torch.empty((samples, steps))

    for t in range(steps):
        a = policy.act(n)[0]
        r, o, s1 = env.step(s, a)
        n1 = policy.step(n, o)[0]

        rewards[:, t] = r

        s, n = s1, n1

    returns = torch.bmm(
        r2g.expand(samples, -1, -1),
        rewards.unsqueeze(-1)
    ).squeeze(-1)
    rets = returns[:, 0]

    return (f'Evaluation {label};\t'
            f'Return: {rets.mean():.2f} / {rets.std():.2f}')

        # print(f'Evaluation {label};\t'
        #       # f'Time steps: {steps};\t'
        #       # f'Reward: {rewards.mean():.2f} / {rewards.std():.2f};\t'
        #       f'Return: {rets.mean():.2f} / {rets.std():.2f}')


def vanilla(env, policy, optimizer, memreplay, episodes, samples, steps):
    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]

    for idx_episode in range(episodes):
        states = torch.empty((samples, steps)).long()
        nodes = torch.empty((samples, steps)).long()
        actions = torch.empty((samples, steps)).long()
        observations = torch.empty((samples, steps)).long()
        rewards = torch.empty((samples, steps))
        anlls = torch.empty((samples, steps))
        nnlls = torch.empty((samples, steps))

        s = env.new((samples,))
        n, nnll = policy.new((samples,))
        for t in range(steps):
            nnlls[:, t] = nnll

            a, anll = policy.act(n)
            r, o, s1 = env.step(s, a)
            n1, nnll = policy.step(n, o)

            states[:, t] = s
            nodes[:, t] = n
            actions[:, t] = a
            observations[:, t] = o
            rewards[:, t] = r
            anlls[:, t] = anll

            s, n = s1, n1

        returns = torch.bmm(
            r2g.expand(samples, -1, -1),
            rewards.unsqueeze(-1)
        ).squeeze(-1)
        returns0 = returns[:, 0]

        memreplay.push(nodes=nodes, actions=actions, observations=observations,
                       rewards=rewards, returns=returns, anlls=anlls.detach(),
                       nnlls=nnlls.detach())

        losses = ((anlls + nnlls) * discounts * returns).sum(1)

        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.ml.parameters(),
                                         config.clip, 'inf')
        optimizer.step()

        if idx_episode % config.log_interval == 0:
            print(f'Episode {idx_episode};\tTime steps: {steps};\t'
                  # f'GNorm: {gnorm:.2e};\t'
                  # f'Loss: {losses.mean():.2f} / {losses.std():.2f};\t'
                  f'Reward: {rewards.mean():.2f} / {rewards.std():.2f};\t'
                  f'Return: {returns0.mean():.2f} / {returns0.std():.2f}')


def baseline(env, policy, optimizer, memreplay, episodes, samples, steps):
    # TODO better way of selecting what to train... whether actor only or critic only or what..

    # TODO I still need separate optimizers..

    criterion = nn.MSELoss(reduce=False)

    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]
    idiscounts = r2g[:, -1]

    for idx_episode in range(episodes):
        states = torch.empty((samples, steps)).long()
        nodes = torch.empty((samples, steps)).long()
        actions = torch.empty((samples, steps)).long()
        observations = torch.empty((samples, steps)).long()
        rewards = torch.empty((samples, steps))
        baselines = torch.empty((samples, steps))
        anlls = torch.empty((samples, steps))
        nnlls = torch.empty((samples, steps))

        s = env.new((samples,))
        n, nnll = policy.new((samples,))
        for t in range(steps):
            nnlls[:, t] = nnll

            a, anll, b = policy.act(n)
            r, o, s1 = env.step(s, a)
            n1, nnll, b1 = policy.step(n, o)

            states[:, t] = s
            nodes[:, t] = n
            actions[:, t] = a
            observations[:, t] = o
            rewards[:, t] = r
            baselines[:, t] = b
            anlls[:, t] = anll

            s, n = s1, n1

        returns = torch.bmm(
            r2g.expand(samples, -1, -1),
            rewards.unsqueeze(-1)
        ).squeeze(-1)
        returns0 = returns[:, 0]

        memreplay.push(nodes=nodes, actions=actions, observations=observations,
                       rewards=rewards, returns=returns, anlls=anlls.detach(),
                       nnlls=nnlls.detach())

        # NOTE Still need the bootstrapping at the end;  It may be negligible
        # for the first returns, but it is not for the last ones.
        targets = returns + env.gamma * b1.ger(idiscounts)
        deltas = targets - baselines
        losses = ((anlls + nnlls) * discounts * deltas.detach()).sum(1)
        # losses = ((anlls + nnlls) * discounts * deltas.detach()).sum(1) \
        #     + criterion(baselines, targets.detach()).mean(1)

        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.ml.parameters(),
                                         config.clip, 'inf')
        optimizer.step()

        if idx_episode % config.log_interval == 0:
            print(f'Episode {idx_episode};\tTime steps: {steps};\t'
                  f'GNorm: {gnorm:.2e};\t'
                  f'Loss: {losses.mean():.2f} / {losses.std():.2f};\t'
                  f'Reward: {rewards.mean():.2f} / {rewards.std():.2f};\t'
                  f'Return: {returns0.mean():.2f} / {returns0.std():.2f}')


def memtrain_vanilla(env, policy, optimizer, memreplay, steps):
    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]

    # batch_size too large and it won't work anymore
    batch_size = 64
    dloader = DataLoader(memreplay, batch_size=batch_size, shuffle=True)
    for idx_batch, data in enumerate(dloader):
        mem_actions = data['actions']
        mem_observations = data['observations']
        mem_rewards = data['rewards']
        mem_returns = data['returns']
        mem_anlls = data['anlls']
        mem_nnlls = data['nnlls']

        #  One new simulation per memory
        samples = len(mem_actions)

        anlls = torch.empty((samples, steps))
        nnlls = torch.empty((samples, steps))

        n, nnll = policy.new((samples,))
        for t in range(steps):
            nnlls[:, t] = nnll

            a = mem_actions[:, t]
            anll = policy.anll(n, a)
            r = mem_rewards[:, t]
            o = mem_observations[:, t]
            n1, nnll = policy.step(n, o)

            anlls[:, t] = anll

            n = n1

        iweights = mem_anlls - anlls.detach()
        # losses = ((anlls + nnlls) * discounts * mem_returns).sum(1)
        # losses = (iweights.exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)
        # losses = (iweights.sum(1, keepdim=True).exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)
        losses = (iweights.cumsum(1).exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)
        # import ipdb;  ipdb.set_trace()
        # losses = (iweights.cumsum(1).exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)

        # print((mem_anlls.detach() - anlls).sum(1).exp())
        # losses = (torch.exp(mem_anlls - anlls).detach() * (anlls + nnlls) * discounts * mem_returns).sum(1)

        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.ml.parameters(),
                                         config.clip, 'inf')
        optimizer.step()

    rewards = mem_rewards
    returns0 = mem_returns[:, 0]

    # print(f'Episode;\tTime steps: {steps};\t'
    #       f'GNorm: {gnorm:.2e};\t'
    #       f'Loss: {losses.mean():.2f} / {losses.std():.2f};\t'
    #       f'Reward: {rewards.mean():.2f} / {rewards.std():.2f};\t'
    #       f'Return: {returns0.mean():.2f} / {returns0.std():.2f}')


def memtrain_baseline(env, policy, optimizer, memreplay, steps):
    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]

    # batch_size too large and it won't work anymore
    batch_size = 64
    dloader = DataLoader(memreplay, batch_size=batch_size, shuffle=True)
    for idx_batch, data in enumerate(dloader):
        mem_actions = data['actions']
        mem_observations = data['observations']
        mem_rewards = data['rewards']
        mem_returns = data['returns']
        mem_anlls = data['anlls']
        mem_nnlls = data['nnlls']

        #  One new simulation per memory
        samples = len(mem_actions)

        anlls = torch.empty((samples, steps))
        nnlls = torch.empty((samples, steps))

        n, nnll = policy.new((samples,))
        for t in range(steps):
            nnlls[:, t] = nnll

            a = mem_actions[:, t]
            anll = policy.anll(n, a)
            r = mem_rewards[:, t]
            o = mem_observations[:, t]
            n1, nnll = policy.step(n, o)

            anlls[:, t] = anll

            n = n1

        iweights = mem_anlls - anlls.detach()
        # losses = ((anlls + nnlls) * discounts * mem_returns).sum(1)
        # losses = (iweights.exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)
        # losses = (iweights.sum(1, keepdim=True).exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)
        losses = (iweights.cumsum(1).exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)
        # import ipdb;  ipdb.set_trace()
        # losses = (iweights.cumsum(1).exp() * (anlls + nnlls) * discounts * mem_returns).sum(1)

        # print((mem_anlls.detach() - anlls).sum(1).exp())
        # losses = (torch.exp(mem_anlls - anlls).detach() * (anlls + nnlls) * discounts * mem_returns).sum(1)

        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.ml.parameters(),
                                         config.clip, 'inf')
        optimizer.step()

    rewards = mem_rewards
    returns0 = mem_returns[:, 0]

    # print(f'Episode;\tTime steps: {steps};\t'
    #       f'GNorm: {gnorm:.2e};\t'
    #       f'Loss: {losses.mean():.2f} / {losses.std():.2f};\t'
    #       f'Reward: {rewards.mean():.2f} / {rewards.std():.2f};\t'
    #       f'Return: {returns0.mean():.2f} / {returns0.std():.2f}')


def memtrain_baseline_criticonly(env, policy, optimizer, memreplay, steps):
    # print('new memtrain phase', len(memreplay))
    criterion = nn.MSELoss(reduce=False)

    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]
    idiscounts = r2g[:, -1]

    # batch_size too large and it won't work anymore
    # batch_size = 64
    batch_size = 10
    dloader = DataLoader(memreplay, batch_size=batch_size, shuffle=True)
    # TODO maybe create a new dataset?...
    for idx_batch, data in enumerate(dloader):
        mem_actions = data['actions']
        mem_observations = data['observations']
        mem_rewards = data['rewards']
        mem_returns = data['returns']
        mem_anlls = data['anlls']
        mem_nnlls = data['nnlls']

        #  One new simulation per memory
        samples = len(mem_actions)

        baselines = torch.empty((samples, steps))
        anlls = torch.empty((samples, steps))
        nnlls = torch.empty((samples, steps))

        n, nnll = policy.new((samples,))
        for t in range(steps):
            nnlls[:, t] = nnll

            a = mem_actions[:, t]
            anll = policy.anll(n, a)
            b = policy.value(n)
            r = mem_rewards[:, t]
            o = mem_observations[:, t]
            n1, nnll, b1 = policy.step(n, o)

            baselines[:, t] = b
            anlls[:, t] = anll

            n = n1

        targets = mem_returns + env.gamma * b1.ger(idiscounts)
        losses = criterion(baselines, targets.detach()).mean(1)

        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.ml.parameters(),
                                         config.clip, 'inf')
        optimizer.step()

    rewards = mem_rewards
    returns0 = mem_returns[:, 0]

    # print(f'Episode;\tTime steps: {steps};\t'
    #       f'GNorm: {gnorm:.2e};\t'
    #       f'Loss: {losses.mean():.2f} / {losses.std():.2f}')


class MemoryDataset(Dataset):
    def __init__(self, maxdata, data=[]):
        self.maxdata = maxdata
        self.data = data

    def push(self, nodes, actions, observations, rewards, returns, anlls, nnlls):
        for n, a, o, r, g, anll, nnll in zip(nodes, actions, observations, rewards, returns, anlls, nnlls):
            self.data.append(dict(nodes=n, actions=a, observations=o, rewards=r, returns=g, anlls=anll, nnlls=nnll))

            if len(self.data) > self.maxdata:
                self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def factory_policy(config):
    # TODO parse fsc string stuff
    # TODO implement File Controller
    # TODO implement Structured Controller

    # NOTE FSC
    # astrat = policies.AStrategy(nnodes, env.nactions, gain=config.gain)
    # ostrat = policies.OStrategy(nnodes, env.nobs, gain=config.gain)

    # NOTE FSC_Sparse
    # K = nnodes / 2
    # astrat = policies.AStrategy(nnodes, env.nactions, gain=config.gain)
    # ostrat = policies.OStrategy_Sparse(nnodes, env.nobs, K, gain=config.gain)

    # NOTE FSC_Reactive
    # K = 3
    # ostrat = policies.OStrategy_Reactive(env.nobs, K)
    # astrat = policies.AStrategy(ostrat.nnodes, env.nactions, gain=config.gain)

    config.critic = config.algo != 'vanilla'
    policy = policies.factory(env, config, config.policy)
    return policy

def factory_optimizer(policy, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(policy.parameters(config),
                              lr=config.lr, momentum=config.momentum)
    elif config.optim == 'adam':
        optimizer = optim.Adam(policy.parameters(config), lr=config.lr)
    elif config.optim == 'adamax':
        optimizer = optim.Adamax(policy.parameters(config), lr=config.lr)

    return optimizer


def factory_algo(config):
    if config.algo == 'vanilla':
        algo = vanilla
        memtrain = memtrain_vanilla
    elif config.algo == 'baseline':
        algo = baseline
        memtrain = memtrain_baseline
        memtrain = memtrain_baseline_criticonly
    elif config.algo == 'actorcritic':
        algo = actorcritic
        memtrain = memtrain_actorcritic
    elif config.algo == 'acl':
        algo = functools.partial(acl, l=config.l)
        memtrain = functools.partial(memtrain_acl, l=config.l)

    return algo, memtrain


if __name__ == '__main__':
    print('config:', config)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    env = pomdp.Environment.from_fname(config.env)

    policy = factory_policy(config)
    algo, memtrain = factory_algo(config)

    memreplay = MemoryDataset(1000)

    msg = evaluate('initial', env, policy, samples=10000, steps=config.steps)
    # print(msg)

    # TODO maybe baseline helps, because it tells us further which memories to
    # ignore...?

    experience = 0
    optimizer_real = factory_optimizer(policy, config)
    for idx_epoch in range(config.epochs):
        algo(env, policy, optimizer_real, memreplay,
                      1, config.samples, config.steps)
                      # config.episodes, config.samples, config.steps)
        if config.noreplay:
            msg = evaluate(f'epoch {idx_epoch} post-train', env, policy, samples=1000, steps=config.steps)
        experience += config.samples

        if not config.noreplay:
            optimizer = factory_optimizer(policy, config)
            for i in range(10):
                memtrain(env, policy, optimizer, memreplay, steps=config.steps)
            msg = evaluate(f'epoch {idx_epoch} post-mem', env, policy, samples=1000, steps=config.steps)

        print(experience)
        print(msg)


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

import rl.pomdp as pomdp


parser = argparse.ArgumentParser(description='PyTorch FSC example')

parser.add_argument('--seed', type=int, default=None)

parser.add_argument('--algo', type=str,
                    choices=['vanilla', 'baseline', 'actorcritic'],
                    default='vanilla')

parser.add_argument('--samples', type=int, default=1)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--steps', type=int, default=100)

parser.add_argument('--optim', type=str, choices=['sgd', 'adam', 'adamax'],
                    default='sgd')
parser.add_argument('--lr', type=float, default=1.)
parser.add_argument('--lra', type=float, default=None)
parser.add_argument('--lrb', type=float, default=None)
parser.add_argument('--lrc', type=float, default=None)
parser.add_argument('--clip', type=float, default='inf')


parser.add_argument('--gtype', type=str,
                    choices=['episodic', 'discounted', 'longterm'],
                    default='longterm')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

config = parser.parse_args()

nnodes, env = 2, pomdp.Environment.from_fname('loadunload.pomdp')
# nnodes, env = 10, pomdp.Environment.from_fname('tiger.pomdp')
# nnodes, env = 20, pomdp.Environment.from_fname('heavenhell.pomdp')
# nnodes, env = 20, pomdp.Environment.from_fname('tag_avoid.pomdp')


def make_r2g(n, gamma):
    r2g = gamma ** (-np.subtract.outer(range(n), range(n)))
    tril_idx = np.tril_indices(n, -1)
    r2g[tril_idx] = 0.
    return torch.from_numpy(r2g.astype(np.float32))


class Identity(nn.Module):
    def forward(Self, inputs):
        return inputs


class MultiEmbedding_from_pretrained(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        idims = torch.tensor(embeddings.shape[:-1])
        odim = embeddings.shape[-1]

        self.midx = idims.cumprod(dim=0).unsqueeze(dim=0) / idims
        embeddings_ = embeddings.reshape((-1, odim))
        self.embedding = nn.Embedding.from_pretrained(embeddings_)

    def forward(self, *codes):
        code = torch.stack(codes, dim=-1)
        code = nnf.linear(code, self.midx).squeeze(dim=-1)
        return self.embedding(code)


class AStrategy(nn.Module):
    def __init__(self, nnodes, nactions, gain=1.):
        super().__init__()
        embeddings = torch.eye(nnodes)
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.linear = nn.Linear(nnodes, nactions, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain)

    def forward(self, n):
        em = self.embedding(n)
        scores = self.linear(em)
        return nnf.softmax(scores, dim=-1)


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
        return nnf.softmax(scores, dim=-1)


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


class OStrategy_Masked(nn.Module):
    def __init__(self, nnodes, nobs, K, gain=1.):
        super().__init__()
        embeddings = torch.eye(nnodes * nobs).reshape(nnodes, nobs, -1)
        self.embedding = MultiEmbedding_from_pretrained(embeddings=embeddings)
        self.linear = nn.Linear(nnodes * nobs, nnodes, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain)

        # TODO create membeddings
        membeddings = torch.where(
            make_mask(nnodes, nobs, K),
            torch.tensor(0.),
            torch.tensor(-float('inf'))
        )
        self.membedding = MultiEmbedding_from_pretrained(
            embeddings=membeddings)

    def forward(self, n, o):
        em = self.embedding(n, o)
        scores = self.linear(em)

        # TODO maybe initialization should also take the mask into acocunt
        mscores = self.membedding(n, o)
        return nnf.softmax(scores + mscores, dim=-1)


class Value(nn.Module):
    def __init__(self, nnodes, gain=1.):
        super().__init__()
        embeddings = torch.eye(nnodes)
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.affine = nn.Linear(nnodes, 1, bias=True)
        nn.init.xavier_normal_(self.affine.weight, .5)
        nn.init.constant_(self.affine.bias, 0)

    def forward(self, n):
        em = self.embedding(n)
        return self.affine(em)


class FSC:
    def __init__(self, env, nnodes, wcritic=False):
        super().__init__()
        self.env = env
        # self.nspace = indextools.RangeSpace(nnodes)

        # self._nshare = Identity()
        self.astrat = AStrategy(nnodes, env.nactions, .9)
        self.ostrat = OStrategy(nnodes, env.nobs, .9)

        self.modules = nn.ModuleList()
        # self.modules.add_module('nshare', self._nshare)
        self.modules.add_module('astrat', self.astrat)
        self.modules.add_module('ostrat', self.ostrat)

        if wcritic:
            self.critic = Value(nnodes)
            self.modules.add_module('critic', self.critic)
        else:
            self.critic = None

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
                pdict['lr'] = config.lrb
            parameters.append(pdict)

        return parameters

    def new(self):
        return torch.tensor(0)

    def act(self, n):
        probs = self.astrat(n)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(n)

        return sample, nll

    def step(self, n, o):
        probs = self.ostrat(n, o)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(sample)

        return sample, nll


class FSC_Sparse:
    def __init__(self, env, nnodes, K, wcritic=False):
        super().__init__()
        self.env = env

        # self._nshare = Identity()
        self.astrat = AStrategy(nnodes, env.nactions, gain=.9)
        # self.ostrat = OStrategy(nnodes, env.nobs, gain=.9)
        self.ostrat = OStrategy_Masked(nnodes, env.nobs, K, gain=.9)

        self.modules = nn.ModuleList()
        # self.modules.add_module('nshare', self._nshare)
        self.modules.add_module('astrat', self.astrat)
        self.modules.add_module('ostrat', self.ostrat)

        if wcritic:
            self.critic = Value(nnodes)
            self.modules.add_module('critic', self.critic)

    def new(self):
        return torch.tensor(0)

    def act(self, n):
        probs = self.astrat(n)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(n)

        return sample, nll

    def step(self, n, o):
        probs = self.ostrat(n, o)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(sample)

        return sample, nll


class Reactive:
    def __init__(self, env, wcritic=False):
        super().__init__()
        self.env = env

        self.nnodes = env.nobs + 1
        self.astrat = AStrategy(self.nnodes, env.nactions, gain=.9)

        self.modules = nn.ModuleList()
        self.modules.add_module('astrat', self.astrat)

        if critic:
            self.critic = Value(self.nnodes)
            self.modules.add_module('critic', self.critic)

    def new(self):
        return torch.tensor(0)

    def act(self, n):
        probs = self.astrat(n)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(n)

        return sample, nll

    def step(self, n, o):
        n1 = o + 1

        if self.critic:
            return n1, 0., self.critic(n1)

        return n1, 0.


class FSC_Reactive:
    def __init__(self, env, K, wcritic=False):
        super().__init__()
        self.env = env
        self.K = K

        self.no = self.env.nobs
        self.mod = self.no ** (K - 1)
        bases = torch.full((K,), self.no).long().cumprod(0).cumsum(0)
        bases.div_(self.no)
        self.bases = torch.cat([
            torch.tensor([0]),
            torch.full((K,), self.no).long().cumprod(0).cumsum(0) / self.no,
        ])

        self.nnodes = self.bases[-1] + self.no ** K
        self.astrat = AStrategy(self.nnodes, env.nactions, gain=.9)

        self.modules = nn.ModuleList()
        self.modules.add_module('astrat', self.astrat)

        if critic:
            self.critic = Value(self.nnodes)
            self.modules.add_module('critic', self.critic)

    def new(self):
        return torch.tensor(0)

    def act(self, n):
        probs = self.astrat(n)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)

        if self.critic:
            return sample, nll, self.critic(n)

        return sample, nll

    def step(self, n, o):
        ibase = (self.bases <= n).sum() - 1
        base = self.bases[ibase]
        try:
            base1 = self.bases[ibase + 1]
        except IndexError:
            base1 = base
        # rule for k-order reactive internal dynamics
        n1 = ((n - base) % self.mod) * self.no + base1 + o

        if self.critic:
            return n1, 0., self.critic(n1)

        return n1, 0.


def show_param_state(policy, optimizer):
    for name, p in policy.modules.named_parameters():
        print('---')
        print('name', name)
        state = optimizer.state[p]
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        # print(exp_avg)
        # print(exp_avg_sq)
        # TODO note, one pair is for astrat and another for...
        print('m1 =', exp_avg.mean())
        print('m2 =', exp_avg_sq.mean())


def vanilla(env, policy):
    r2g = make_r2g(config.steps, env.gamma)
    discounts = r2g[0]

    for idx_episode in range(config.episodes):
        rets = torch.empty(config.samples)
        losses = torch.empty(config.samples)

        for idx_sample in range(config.samples):
            s = env.new()
            n = policy.new()

            nlls = torch.empty(config.steps)
            rewards = torch.empty(config.steps)

            nll_t = 0.
            for t in range(config.steps):
                a, nll_a = policy.act(n)
                r, o, s1 = env.step(s, a)
                n1, nll_n = policy.step(n, o)

                nll_t += nll_a
                nlls[t] = nll_t
                nll_t = nll_n

                rewards[t] = r

                s, n = s1, n1

            returns = torch.mv(r2g, rewards)
            rets[idx_sample] = returns[0]
            losses[idx_sample] = (nlls * discounts * returns).sum()

        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.modules.parameters(),
                                         config.clip, 'inf')
        optimizer.step()

        if idx_episode % config.log_interval == 0:
            # show_param_state(policy, optimizer)

            print('gnorm', gnorm)
            print(f'Episode {idx_episode};\tTime steps: {config.steps};\t'
                  f'Loss: {losses.mean():.2f} / {losses.std():.2f};\t'
                  f'Return: {rets.mean():.2f} / {rets.std():.2f}')


def baseline(env, policy):
    criterion = nn.MSELoss(reduce=False)

    r2g = make_r2g(config.steps, env.gamma)
    discounts = r2g[0]

    for idx_episode in range(config.episodes):
        rets = torch.empty(config.samples)
        losses = torch.empty(config.samples)

        for idx_sample in range(config.samples):
            s = env.new()
            n = policy.new()

            nlls = torch.empty(config.steps)
            rewards = torch.empty(config.steps)
            baselines = torch.empty(config.steps)

            nll_t = 0.
            for t in range(config.steps):
                a, nll_a, b = policy.act(n)
                r, o, s1 = env.step(s, a)
                n1, nll_n, _ = policy.step(n, o)

                nll_t += nll_a
                nlls[t] = nll_t
                nll_t = nll_n

                rewards[t] = r
                baselines[t] = b

                s, n = s1, n1

            returns = torch.mv(r2g, rewards)
            rets[idx_sample] = returns[0]

            target = returns
            delta = target - baselines
            losses[idx_sample] = (nlls * discounts * delta.detach()).sum() \
                + criterion(baselines, target.detach()).mean()
            # losses[idx_sample] = (nlls * discounts * delta.detach()).sum() \
            #     + (discounts * criterion(baselines, target.detach())).mean()

        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.modules.parameters(),
                                         config.clip, 'inf')
        optimizer.step()

        if idx_episode % config.log_interval == 0:
            # show_param_state(policy, optimizer)
            # print(delta)

            print('gnorm', gnorm)
            print(f'Episode {idx_episode};\tTime steps: {config.steps};\t'
                  f'Loss: {losses.mean():.2f} / {losses.std():.2f};\t'
                  f'Return: {rets.mean():.2f} / {rets.std():.2f}')


def actorcritic(env, policy):
    criterion = nn.MSELoss(reduce=False)

    r2g = make_r2g(config.steps, env.gamma)
    discounts = r2g[0]

    for idx_episode in range(config.episodes):
        rets = torch.empty(config.samples)
        losses = torch.empty(config.samples)

        for idx_sample in range(config.samples):
            s = env.new()
            n = policy.new()

            nlls = torch.empty(config.steps)
            rewards = torch.empty(config.steps)
            baselines = torch.empty(config.steps)
            baselines1 = torch.empty(config.steps)

            nll_t = 0.
            for t in range(config.steps):
                a, nll_a, b = policy.act(n)
                r, o, s1 = env.step(s, a)
                n1, nll_n, b1 = policy.step(n, o)

                nll_t += nll_a
                nlls[t] = nll_t
                nll_t = nll_n

                rewards[t] = r
                baselines[t] = b
                baselines1[t] = b1

                s, n = s1, n1

            returns = torch.mv(r2g, rewards)
            rets[idx_sample] = returns[0]

            # print(baselines)
            target = rewards + env.gamma * baselines1
            delta = target - baselines
            losses[idx_sample] = (nlls * discounts * delta.detach()).sum() \
                + criterion(baselines, target.detach()).mean()
            # losses[idx_sample] = (nlls * discounts * delta.detach()).sum() \
            #     + (discounts * criterion(baselines, target.detach())).mean()
            # losses[idx_sample] = (nlls * discounts * delta.detach()).sum() \
            #     + criterion(baselines - env.gamma * baselines1,
            #                 rewards).mean()
            # losses[idx_sample] = (nlls * discounts * delta.detach()).sum() \
            #     + (discounts * criterion(baselines - env.gamma * baselines1,
            #                              rewards)).mean()

            # TODO how to treat clipping of multiple gradients things..

        # TODO is the baselines / target weighting correct?
        # TODO supposedly, since per-dimension clipping is a good idea,
        # why not per dimension step size?
        # Per-dimension clipping is probably a bad idea..
        # TODO maybe the actor-critic version should be one-step version?
        # that could make sense!  I only need one step at the time anyway...
        # TODO try inf norm gradient

        # print(baselines)
        # print(baselines1)
        # print(baselines - env.gamma * baselines1)
        optimizer.zero_grad()
        losses.mean().backward()
        gnorm = nn.utils.clip_grad_norm_(policy.modules.parameters(),
                                         config.clip, 'inf')
        # print(gnorm)
        # print(baselines)
        optimizer.step()

        if idx_episode % config.log_interval == 0:
            # print(baselines)
            # print(delta)
            print('gnorm', gnorm)
            print(f'Episode {idx_episode};\tTime steps: {config.steps};\t'
                  f'Loss: {losses.mean():.2f} / {losses.std():.2f};\t'
                  f'Return: {rets.mean():.2f} / {rets.std():.2f}')


def algo(env, policy, algo):
    if algo == 'vanilla':
        vanilla(env, policy)
    elif algo == 'baseline':
        baseline(env, policy)
    elif algo == 'actorcritic':
        actorcritic(env, policy)


if __name__ == '__main__':
    print(config)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    K = 3
    if config.algo == 'vanilla':
        # policy = Reactive(env)
        policy = FSC(env, nnodes)
        # policy = FSC_Sparse(env, nnodes, 2)
        # policy = FSC_Reactive(env, K)
    else:
        # policy = Reactive(env, wcritic=True)
        policy = FSC(env, nnodes, wcritic=True)
        # policy = FSC_Sparse(env, nnodes, 2, wcritic=True)
        # policy = FSC_Reactive(env, K, wcritic=True)

    if config.optim == 'sgd':
        optimizer = optim.SGD(policy.parameters(config), lr=config.lr)
    elif config.optim == 'adam':
        optimizer = optim.Adam(policy.parameters(config), lr=config.lr)
    elif config.optim == 'adamax':
        optimizer = optim.Adamax(policy.parameters(config), lr=config.lr)

    algo(env, policy, config.algo)

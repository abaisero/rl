#!/usr/bin/env python
import sys
import argparse
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as tmp

import numpy as np

import rl.pomdp as pomdp
import rl.pomdp.policies as policies


parser = argparse.ArgumentParser(description='PyTorch FSC example')

parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--out', type=str, default=None)

parser.add_argument('--device', type=torch.device,
                    default='cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument('env', type=str, help='environment')
# parser.add_argument('policy', type=str, help='policy')
parser.add_argument('algo', type=str,
                    choices=['vanilla', 'baseline', 'actorcritic', 'acl'],
                    default='vanilla')

parser.add_argument('--l', type=float)

parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--episodes', type=int, default=500)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--samples', type=int, default=1)

parser.add_argument('--optim', type=str, choices=['sgd', 'adam', 'adamax'],
                    default='sgd')
parser.add_argument('--lr', type=float, default=1.)
parser.add_argument('--lra', type=float, default=None)
parser.add_argument('--lrb', type=float, default=None)
parser.add_argument('--lrc', type=float, default=None)
parser.add_argument('--momentum', type=float, default=0.)
parser.add_argument('--clip', type=float, default='inf')

parser.add_argument('--gtype', type=str,
                    choices=['episodic', 'discounted', 'longterm'],
                    default='longterm')

parser.add_argument('--gain', type=float, default=1.)

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

config = parser.parse_args()

def make_r2g(n, v):
    r2g = v ** (-np.subtract.outer(range(n), range(n)))
    tril_idx = np.tril_indices(n, -1)
    r2g[tril_idx] = 0.
    return torch.from_numpy(r2g.astype(np.float32)).to(config.device)


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

#     def step(self, n, o):
#         probs = self.ostrat(n, o)
#         dist = Categorical(probs)
#         sample = dist.sample()
#         nll = -dist.log_prob(sample)

#         if self.critic:
#             return sample, nll, self.critic(sample).squeeze(-1)

#         return sample, nll


# def show_param_state(policy, optimizer):
#     for name, p in policy.ml.named_parameters():
#         print('---')
#         print('name', name)
#         state = optimizer.state[p]
#         exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#         # print(exp_avg)
#         # print(exp_avg_sq)
#         # TODO note, one pair is for astrat and another for...
#         print('m1 =', exp_avg.mean())
#         print('m2 =', exp_avg_sq.mean())


def vanilla(env, policy, optimizer, episodes, samples, steps, device):
    rews = torch.empty((episodes, samples, steps)).to(device)
    rets = torch.empty((episodes, samples)).to(device)

    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]

    for idx_episode in range(episodes):
        rewards = torch.empty((samples, steps)).to(device)
        anlls = torch.empty((samples, steps)).to(device)
        nnlls = torch.empty((samples, steps)).to(device)

        s = env.new((samples,), device=device)
        n, nnll = policy.new((samples,), device=device)
        for t in range(steps):
            nnlls[:, t] = nnll

            a, anll = policy.act(n)
            r, o, s1 = env.step(s, a)
            n1, nnll = policy.step(n, o)

            rewards[:, t] = r
            anlls[:, t] = anll

            s, n = s1, n1

        returns = torch.bmm(
            r2g.expand(samples, -1, -1),
            rewards.unsqueeze(-1)
        ).squeeze(-1)
        returns0 = returns[:, 0]

        rews[idx_episode] = rewards
        rets[idx_episode] = returns0

        losses = ((anlls + nnlls) * discounts * returns).sum(1)

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

    return rews, rets


def baseline(env, policy, optimizer, episodes, samples, steps, device):
    rews = torch.empty((episodes, samples, steps)).to(device)
    rets = torch.empty((episodes, samples)).to(device)

    criterion = nn.MSELoss(reduce=False)

    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]
    idiscounts = r2g[:, -1]

    for idx_episode in range(episodes):
        rewards = torch.empty((samples, steps)).to(device)
        baselines = torch.empty((samples, steps)).to(device)
        anlls = torch.empty((samples, steps)).to(device)
        nnlls = torch.empty((samples, steps)).to(device)

        s = env.new((samples,), device=device)
        n, nnll = policy.new((samples,), device=device)
        for t in range(steps):
            nnlls[:, t] = nnll

            a, anll, b = policy.act(n)
            r, o, s1 = env.step(s, a)
            n1, nnll, b1 = policy.step(n, o)

            rewards[:, t] = r
            baselines[:, t] = b
            anlls[:, t] = anll

            s, n = s1, n1

        returns = torch.bmm(
            r2g.expand(samples, -1, -1),
            rewards.unsqueeze(-1)
        ).squeeze(-1)
        returns0 = returns[:, 0]

        rews[idx_episode] = rewards
        rets[idx_episode] = returns0

        # NOTE Still need the bootstrapping at the end;  It may be negligible
        # for the first returns, but it is not for the last ones.
        targets = returns + env.gamma * b1.ger(idiscounts)
        deltas = targets - baselines
        losses = ((anlls + nnlls) * discounts * deltas.detach()).sum(1) \
            + criterion(baselines, targets.detach()).mean(1)

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

    return rews, rets


def actorcritic(env, policy, optimizer, episodes, samples, steps, device):
    rews = torch.empty((episodes, samples, steps)).to(device)
    rets = torch.empty((episodes, samples)).to(device)

    criterion = nn.MSELoss(reduce=False)

    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]

    for idx_episode in range(episodes):
        rewards = torch.empty((samples, steps)).to(device)
        baselines = torch.empty((samples, steps)).to(device)
        baselines1 = torch.empty((samples, steps)).to(device)
        anlls = torch.empty((samples, steps)).to(device)
        nnlls = torch.empty((samples, steps)).to(device)

        s = env.new((samples,), device=device)
        n, nnll = policy.new((samples,), device=device)
        for t in range(steps):
            nnlls[:, t] = nnll

            a, anll, b = policy.act(n)
            r, o, s1 = env.step(s, a)
            n1, nnll, b1 = policy.step(n, o)

            rewards[:, t] = r
            baselines[:, t] = b
            baselines1[:, t] = b1
            anlls[:, t] = anll

            s, n = s1, n1

        returns = torch.bmm(
            r2g.expand(samples, -1, -1),
            rewards.unsqueeze(-1)
        ).squeeze(-1)
        returns0 = returns[:, 0]

        rews[idx_episode] = rewards
        rets[idx_episode] = returns0

        targets = rewards + env.gamma * baselines1
        deltas = targets - baselines
        losses = ((anlls + nnlls) * discounts * deltas.detach()).sum(1) \
            + criterion(baselines, targets.detach()).mean(1)

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

    return rews, returns


def acl(env, policy, optimizer, episodes, samples, steps, device, *, l):
    rews = torch.empty((episodes, samples, steps)).to(device)
    rets = torch.empty((episodes, samples)).to(device)

    criterion = nn.MSELoss(reduce=False)

    r2g = make_r2g(steps, env.gamma)
    discounts = r2g[0]

    A = make_r2g(steps, env.gamma * l)
    B = A * env.gamma * (1. - l)
    c = A[:, -1] * env.gamma * l

    for idx_episode in range(episodes):
        rewards = torch.empty((samples, steps)).to(device)
        baselines = torch.empty((samples, steps)).to(device)
        baselines1 = torch.empty((samples, steps)).to(device)
        anlls = torch.empty((samples, steps)).to(device)
        nnlls = torch.empty((samples, steps)).to(device)

        s = env.new((samples,), device=device)
        n, nnll = policy.new((samples,), device=device)
        for t in range(steps):
            nnlls[:, t] = nnll

            a, anll, b = policy.act(n)
            r, o, s1 = env.step(s, a)
            n1, nnll, b1 = policy.step(n, o)

            rewards[:, t] = r
            baselines[:, t] = b
            baselines1[:, t] = b1
            anlls[:, t] = anll

            s, n = s1, n1

        returns = torch.bmm(
            r2g.expand(samples, -1, -1),
            rewards.unsqueeze(-1)
        ).squeeze(-1)
        returns0 = returns[:, 0]

        lreturns = torch.bmm(
            A.expand(samples, -1, -1),
            rewards.unsqueeze(-1)
        ).squeeze(-1) + torch.bmm(
            B.expand(samples, -1, -1),
            baselines1.unsqueeze(-1),
        ).squeeze(-1) + b1.ger(c)

        rews[idx_episode] = rewards
        rets[idx_episode] = returns0

        # TODO what is the baseline supposed to be? just the standard
        # non-lambda return, methinks.... They are all "equally valid"
        # estimates for the full return double check?
        targets = lreturns
        deltas = targets - baselines
        losses = ((anlls + nnlls) * discounts * deltas.detach()).sum(1) \
            + criterion(baselines, targets.detach()).mean(1)

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

    return rews, returns


def make_policy(config):
    # TODO parse fsc string stuff
    # TODO implement File Controller
    # TODO implement Structured Controller

    # NOTE FSC
    astrat = policies.AStrategy(nnodes, env.nactions, gain=config.gain)
    ostrat = policies.OStrategy(nnodes, env.nobs, gain=config.gain)

    # NOTE FSC_Sparse
    # K = nnodes / 2
    # astrat = policies.AStrategy(nnodes, env.nactions, gain=config.gain)
    # ostrat = policies.OStrategy_Sparse(nnodes, env.nobs, K, gain=config.gain)

    # NOTE FSC_Reactive
    # K = 3
    # ostrat = policies.OStrategy_Reactive(env.nobs, K)
    # astrat = policies.AStrategy(ostrat.nnodes, env.nactions, gain=config.gain)

    critic = policies.Value(nnodes) if config.algo != 'vanilla' else None
    policy = policies.FSC(astrat, ostrat, critic=critic)
    policy.ml.to(config.device)

    if config.optim == 'sgd':
        optimizer = optim.SGD(policy.parameters(config),
                              lr=config.lr, momentum=config.momentum)
    elif config.optim == 'adam':
        optimizer = optim.Adam(policy.parameters(config), lr=config.lr)
    elif config.optim == 'adamax':
        optimizer = optim.Adamax(policy.parameters(config), lr=config.lr)

    return policy, optimizer


def make_algo(config):
    if config.algo == 'vanilla':
        algo = functools.partial(vanilla,
                                 episodes=config.episodes,
                                 samples=config.samples,
                                 steps=config.steps,
                                 device=config.device)
    elif config.algo == 'baseline':
        algo = functools.partial(baseline,
                                 episodes=config.episodes,
                                 samples=config.samples,
                                 steps=config.steps,
                                 device=config.device)
    elif config.algo == 'actorcritic':
        algo = functools.partial(actorcritic,
                                 episodes=config.episodes,
                                 samples=config.samples,
                                 steps=config.steps,
                                 device=config.device)
    elif config.algo == 'acl':
        algo = functools.partial(acl,
                                 episodes=config.episodes,
                                 samples=config.samples,
                                 steps=config.steps,
                                 device=config.device,
                                 l=config.l)

    return algo


if __name__ == '__main__':
    print('config:', config)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    nnodes = 4
    env = pomdp.Environment.from_fname(config.env)

    rewards, returns = [], []
    # TODO just run in different processes?  Maybe not..... and...?
    for idx_run in range(config.runs):
        policy, optimizer = make_policy(config)
        algo = make_algo(config)

        rews, rets = algo(env, policy, optimizer)
        rewards.append(rews)
        returns.append(rets)

    # rewards = torch.stack(rewards).cpu()
    # returns = torch.stack(returns).cpu()

    returns0 = returns[:, 0, :]

    if config.out is not None:
        fname = f'{config.out}.pt'
        torch.save(returns0, fname)

        fname = f'{config.out}.txt'
        args='\n'.join(sys.argv)
        with open(fname, 'w') as f:
            print('# sys argv', file=f)
            print(sys.argv, file=f)
            print(file=f)
            print('# args', file=f)
            print(args, file=f)
            print(file=f)
            print('# config', file=f)
            print(config, file=f)
            print(file=f)

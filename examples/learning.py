#!/usr/bin/env python

import logging.config
from logconfig import LOGGING

import rl.pomdp as pomdp
import rl.pomdp.envs as envs
import rl.pomdp.policies as policies
import rl.pomdp.psearch as psearch
import rl.pomdp.agents as agents
import rl.optim as optim
import rl.graph as graph
# import rl.parsers.dotpomdp as dotpomdp
# import rl.parsers.dotfsc as dotfsc
import rl.misc as misc

from pytk.callbacks import Callback

import numpy as np
import numpy.random as rnd

from pyqtgraph.Qt import QtCore

import multiprocessing as mp
import time


import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='blarg')

    parser.add_argument('--graph', action='store_const', const=True,
            help='show graphics', default=False)
    parser.add_argument('--out', type=str,
            help='output file name', default=None)

    parser.add_argument('--runs', metavar='r', type=int,
            help='number of training runs', default=10)
    parser.add_argument('--episodes', metavar='e', type=int,
            help='number of training episodes', default=1000)
    parser.add_argument('--horizon', metavar='h', type=int,
            help='number of steps in an episode', default=100)
    parser.add_argument('--processes', metavar='p', type=int,
            help='number of processes', default=mp.cpu_count())

    parser.add_argument('pomdp', type=str, help='POMDP name')
    policies.add_subparsers(parser)

    args = parser.parse_args()
    print(args)

    # import sys
    # sys.exit(0)

    nruns = args.runs
    nepisodes = args.episodes
    horizon = args.horizon
    processes = args.processes
    pomdp_name = args.pomdp


    # np.seterr(all='raise')

    # logging configuration
    logging.config.dictConfig(LOGGING)

    # nruns, nepisodes, horizon = 10, 5000, 100

    # NOTE
    # envname, N, beta, step_size = 'loadunload', 2, .95, optim.StepSize(1)

    env = envs.env(pomdp_name)
    # TODO maybe do this?
    # env = pomdp.Environment.from_dotpomdp(pomdp_name)
    # env = envs.Tiger(.01)
    # env.gamma = .95

    # TODO save running results and combine in a table / plot...

    policy_cls = policies.policy_cls(args.policy)
    policy_ns = getattr(args, args.policy)
    policy = policy_cls.from_namespace(env, policy_ns)


#     N = 10
#     K = 3
    beta = .95
    # step_size = optim.StepSize(1)
    step_size = optim.StepSize(.1)
    # step_size = optim.StepSize(.01)
    # step_size = optim.Geometric(10, .999)
    eps = 1e-10
    # # processes = 1
    # # processes = mp.cpu_count()
    # processes = mp.cpu_count() - 1

    # TODO maybe FSC needs different step sizes for different strategies?

    # Random
    # policy = policies.Random(env)
    # agent = agents.Agent('Random', env, policy)

    # TODO blind should also use the policy gradient interface....
    # TODO blind should just be a type of policy...... then just use GPOMDP

    # Blind (tries to learn best action distribution)
    # policy = policies.Blind(env)
    # agent = agents.Blind('Blind', env, policy)

    # GPOMDP
    # policy = policies.Reactive(env)
    # pg = psearch.GPOMDP(policy, beta)
    # name = f'GPOMDP ($\\beta$={beta})'
    # agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # CONJGPOMDP + GPOMDP
    # policy = policies.Reactive(env)
    # pg = psearch.GPOMDP(policy, beta)
    # ps = psearch.CONJPOMDP(policy, pg, step_size=step_size, eps=eps)
    # name = f'CONJPOMDP-GPOMDP ($\\beta$={beta}, $\\epsilon$={eps})'
    # agent = agents.PolicySearch(name, env, policy, ps)

    # Istate-GPOMDP (params N and beta)
    # policy = policies.FSC(env, N)
    # pg = psearch.IsGPOMDP(policy, beta)
    # name = f'IsGPOMDP (N={N}, $\\beta$={beta})'
    # agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # Sparse Istate-GPOMDP (params N, K and beta)
    # policy = policies.SparseFSC(env, N, K)
    # pg = psearch.IsGPOMDP(policy, beta)
    # name = f'IsGPOMDP (N={N}, $\\beta$={beta})'
    # agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # Structured Istate-GPOMDP (params N and beta)
    # fss_name = 'tiger.lin.1.fss'  # early success
    # fss_name = 'tiger.lin.2.fss'  # late success
    # fss_name = 'tiger.lin.3.fss'  #
    # TODO also move this to...
    # policy = dotfsc.fsc(fscname, env)
    # policy = policies.parse_dotfss(fss_name, env)

    pg = psearch.IsGPOMDP(policy, beta)
    name = f'IsGPOMDP ($\\beta$={beta})'
    agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # CONJGPOMDP + Istate-GPOMDP
    # policy = policies.FSC(env, N)
    # pgrad = psearch.IsGPOMDP(policy, beta)
    # ps = psearch.CONJPOMDP(policy, pgrad, step_size=step_size, eps=eps)
    # name = f'CONJPOMDP-IsGPOMDP (N={N}, $\\beta$={beta}, $\\epsilon$={eps})'
    # agent = agents.PolicySearch(name, env, policy, ps)


    if args.graph:
        def pdict_item(percentile, **kwargs):
            return percentile, dict(name=f'{percentile/100:.2f}', **kwargs)

        pdict = dict([
            pdict_item(  0, pen=dict(color='r', style=QtCore.Qt.DotLine)),
            pdict_item(100, pen=dict(color='g', style=QtCore.Qt.DotLine)),
            pdict_item( 25, pen=dict(color='r')),
            pdict_item( 75, pen=dict(color='g')),
            pdict_item( 50),
        ])

        q_returns, _ = graph.pplot((nruns, nepisodes), pdict,
            window=dict(text='Returns', size='16pt', bold=True),
            labels=dict(left='G_t', bottom='Episode'),
        )
        q_gnorms, _ = graph.pplot((nruns, nepisodes), pdict,
            window=dict(text='Gradient Norms', size='16pt', bold=True),
            labels=dict(left='|w|', bottom='Episode'),
        )

        try:
            pplot = policy.plot
        except AttributeError:
            policy_plot = False
        else:
            pplot(nepisodes)
            policy_plot = True

    H = misc.Horizon(horizon)
    sys = pomdp.System(env, env.model, H)

    v = mp.RawValue('i', 0)
    l = mp.Lock()
    def run(ri):
        with l:
            # true sequential index
            i = v.value
            v.value += 1

        seed = int(time.time() * 1000 + i * 61001) % 2 ** 32
        print(f'Starting run {i+1} / {nruns};  Running {nepisodes} episodes... (with seed {seed})')

        rnd.seed(seed)  # ensure different randomization

        idx_offset = i * nepisodes
        idx_returns = 0
        idx_gnorms = 0

        # returns_run = np.empty(nepisodes)
        returns_run = np.zeros(nepisodes)

        # NOTE if this is outside I have a bug
        @Callback
        def episode_return(sys, episode):
            G = 0.
            for context, a, feedback in episode:
                r = feedback.r
                G = r + env.gamma * G
            return G

        feedbacks = ()
        feedbacks_episode = episode_return,

        if args.graph:
            if i == 0:
                if policy_plot is not None:
                    def episode_dplot(sys, episode):
                        policy.plot_update()

                    feedbacks_episode += episode_dplot,

        # TODO I can still print the different between parameters!!!!
        # YES this is correct..
        if args.graph:
            def episode_gnorm(gradient):
                nonlocal idx_gnorms
                if gradient.dtype == object:
                    gnorm = np.sqrt(sum(_.sum() for _ in gradient ** 2))
                else:
                    gnorm = np.sqrt(np.sum(gradient ** 2))
                q_gnorms.put((idx_gnorms, gnorm))
                idx_gnorms += 1

            # TODO better way to handle callbacks...
            agent.callbacks_episode = [episode_gnorm]
            # TODO... how to do callbacks for different types of agents?

            @episode_return.callback
            def plot_return(G):
                nonlocal idx_returns
                q_returns.put((idx_offset + idx_returns, G))
                returns_run.itemset(idx_returns, G)
                idx_returns += 1
        else:
            @episode_return.callback
            def save_return(G):
                nonlocal idx_returns
                returns_run.itemset(idx_returns, G)
                idx_returns += 1

        #  reset agent before each new learning run
        agent.reset()
        sys.run(
            agent,
            nepisodes=nepisodes,
            feedbacks=feedbacks,
            feedbacks_episode=feedbacks_episode,
        )

        return returns_run


    if processes > 1:  # NOTE parallelized
        with mp.Pool(processes=processes) as pool:
            returns = np.array(pool.map(run, range(nruns)))
    else:  # NOTE serialized
        returns = np.array([run(ri) for ri in range(nruns)])

    try:
        np.save(args.out, returns)
    except AttributeError:
        pass

    if args.graph:
        q_returns.put(None)
        q_gnorms.put(None)

        #  keeps graphics alive
        import IPython
        IPython.embed()


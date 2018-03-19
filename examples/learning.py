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
import rl.misc as misc
import rl.core.returns as returns
import rl.misc.blinker as rl_blinker
import blinker

# from pytk.callbacks import Callback

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

    parser.add_argument('--processes', metavar='p', type=int,
            help='number of processes', default=mp.cpu_count())
    parser.add_argument('--runs', metavar='r', type=int,
            help='number of training runs', default=10)
    parser.add_argument('--episodes', metavar='e', type=int,
            help='number of training episodes', default=1000)
    parser.add_argument('--horizon', metavar='h', type=int,
            help='number of steps in an episode', default=100)
    parser.add_argument('--nu', dest='objective',
            action='store_const', const='longterm_average',
            default='longterm_average')
    parser.add_argument('--J', dest='objective',
            action='store_const', const='discounted_sum')

    parser.add_argument('pomdp', type=str, help='POMDP name')
    policies.add_subparsers(parser)

    args = parser.parse_args()
    print(f'Argument Namespace: {args}')

    # logging configuration
    logging.config.dictConfig(LOGGING)

    env = envs.env(args.pomdp)

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
            pdict_item( 25, pen=dict(color='r')),
            pdict_item( 75, pen=dict(color='g')),
            pdict_item(  0, pen=dict(color='r', style=QtCore.Qt.DotLine)),
            pdict_item(100, pen=dict(color='g', style=QtCore.Qt.DotLine)),
            pdict_item( 50),
        ])

        q_returns, _ = graph.pplot((args.runs, args.episodes), pdict,
            window=dict(text='Returns', size='16pt', bold=True),
            labels=dict(left='G_t', bottom='Episode'),
        )
        q_gnorms, _ = graph.pplot((args.runs, args.episodes), pdict,
            window=dict(text='Gradient Norms', size='16pt', bold=True),
            labels=dict(left='|w|', bottom='Episode'),
        )

        try:
            pplot = policy.plot
        except AttributeError:
            policy_plot = False
        else:
            pplot(args.episodes)
            policy_plot = True

    horizon = misc.Horizon(args.horizon)
    sys = pomdp.System(env, env.model, horizon)

    v = mp.RawValue('i', 0)
    l = mp.Lock()

    seed0 = int(time.time() * 1000) % 2**32

    def run(ri):
        with l:
            # true sequential index
            i = v.value
            v.value += 1

        # ensure different randomization for each run
        seed = seed0 + i
        print(f'Starting run {i+1} / {args.runs};  Running {args.episodes} episodes... (with seed {seed})')
        rnd.seed(seed)

        idx_offset = i * args.episodes
        idx_returns = 0
        idx_gnorms = 0

        returns_run = np.empty(args.episodes)
        # returns_run = np.zeros(args.episodes)

        # TODO bug with return plot
        # episode_return = Callback(getattr(returns, args.objective))

        signal_episode_end = blinker.signal('episode-end')
        objective = getattr(returns, args.objective)
        objective = rl_blinker.fire_after(signal_episode_end)(objective)

        # TODO replace all these feedback functions with a single event manager!!!
        # this would also avoid bugs like the "double-step" one
        feedbacks = ()
        # feedbacks_episode = episode_return,
        feedbacks_episode = objective,

        if args.graph:
            if i == 0:
                if policy_plot is not None:
                    def episode_dplot(sys, episode):
                        policy.plot_update()

                    feedbacks_episode += episode_dplot,

        # TODO I can still print the different between parameters!!!!
        if args.graph:
            def episode_gnorm(gradient):
                nonlocal idx_gnorms
                if gradient.dtype == object:
                    gnorm = np.sqrt(sum(_.sum() for _ in gradient ** 2))
                else:
                    gnorm = np.sqrt(np.sum(gradient ** 2))
                q_gnorms.put((idx_offset + idx_gnorms, gnorm))
                idx_gnorms += 1

            # TODO better way to handle callbacks...
            agent.callbacks_episode = [episode_gnorm]
            # TODO... how to do callbacks for different types of agents?

            # @episode_return.callback
            @signal_episode_end.connect
            def plot_return(sender, result):
                nonlocal idx_returns
                q_returns.put((idx_offset + idx_returns, result))
                returns_run.itemset(idx_returns, result)
                idx_returns += 1
        else:
            # @episode_return.callback
            @signal_episode_end.connect
            def save_return(sender, result):
                nonlocal idx_returns
                returns_run.itemset(idx_returns, result)
                idx_returns += 1

        #  reset agent before each new learning run
        agent.reset()
        sys.run(
            agent,
            nepisodes=args.episodes,
            feedbacks=feedbacks,
            feedbacks_episode=feedbacks_episode,
        )

        return returns_run


    run_args = range(args.runs)
    if args.processes > 1:  # parallel
        with mp.Pool(processes=args.processes) as pool:
            rets = pool.map(run, run_args)
    else:  # sequential
        rets = [run(a) for a in run_args]
    rets = np.array(rets)


    try:
        out_fname = args.out
    except AttributeError:
        pass
    finally:
        np.save(out_fname, rets)


    if args.graph:
        q_returns.put(None)
        q_gnorms.put(None)

        #  keeps graphics alive
        import IPython
        IPython.embed()

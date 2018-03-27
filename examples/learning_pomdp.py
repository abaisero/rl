#!/usr/bin/env python

import logging.config
from logconfig import LOGGING

import rl.pomdp as pomdp

import rl.pomdp.objectives as objectives
import rl.pomdp.domains as domains
import rl.pomdp.policies as policies
import rl.pomdp.algos as algos
import rl.optim as optim
import rl.graph as graph
from pyqtgraph.Qt import QtCore

import numpy as np
import numpy.random as rnd

import multiprocessing as mp
import time


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Learning')

    parser.add_argument('--graph', action='store_const', const=True,
            help='show graphics', default=False)
    parser.add_argument('--out', type=str,
            help='output file name', default=None)

    parser.add_argument('--processes', metavar='p', type=int,
            help='number of processes', default=mp.cpu_count()-1)
    parser.add_argument('--runs', type=int,
            help='number of training runs', default=10)
    parser.add_argument('--episodes', type=int,
            help='number of training episodes', default=1000)
    parser.add_argument('--steps', type=int,
            help='number of steps in an episode', default=100)
    parser.add_argument('--nu', dest='obj',
            action='store_const', const='longterm_average',
            default='longterm_average')
    parser.add_argument('--J', dest='obj',
            action='store_const', const='discounted_sum')
    parser.add_argument('pomdp', type=str, help='POMDP name')


    args_ns, rest = parser.parse_known_args()
    algo_ns, rest = algos.parser.parse_known_args(rest)
    policy_ns, rest = policies.parser.parse_known_args(rest)

    return args_ns, algo_ns, policy_ns


if __name__ == '__main__':
    # logging configuration
    logging.config.dictConfig(LOGGING)

    args_ns, algo_ns, policy_ns = parse_arguments()
    print(f'Argument Namespace: {args_ns}')
    print(f'Algorithm Namespace: {algo_ns}')
    print(f'Policy Namespace: {policy_ns}')

    nruns = args_ns.runs
    nepisodes = args_ns.episodes
    nsteps = args_ns.steps
    obj = args_ns.obj

    # TODO check that algo-policy combination works

    domain = pomdp.domains.from_fname(args_ns.pomdp)
    objective = getattr(pomdp.objectives, obj)(domain)


    algo_ns.stepsize = optim.StepSize(.1)  # TODO set using argv
    if algo_ns.beta is None:
        algo_ns.beta = domain.gamma

    # TODO joint factory?!
    algo = algos.factory(algo_ns)
    policy = policies.factory(domain, policy_ns)

    print('POMDP:', domain)
    print('Algo:', algo)
    print('Policy:', policy)

    print(pomdp.env)
    env = pomdp.env.Environment(domain, objective)

    if args_ns.graph:
        def pdict_item(percentile, **kwargs):
            return percentile, dict(name=f'{percentile/100:.2f}', **kwargs)

        pdict = dict([
            pdict_item( 25, pen=dict(color='r')),
            pdict_item( 75, pen=dict(color='g')),
            pdict_item(  0, pen=dict(color='r', style=QtCore.Qt.DotLine)),
            pdict_item(100, pen=dict(color='g', style=QtCore.Qt.DotLine)),
            pdict_item( 50),
        ])

        # TODO better graphing interface
        q_returns, _ = graph.pplot((args_ns.runs, args_ns.episodes), pdict,
            window=dict(text='Returns', size='16pt', bold=True),
            labels=dict(left='G_t', bottom='Episode'),
        )
        q_gnorms, _ = graph.pplot((args_ns.runs, args_ns.episodes), pdict,
            window=dict(text='Gradient Norms', size='16pt', bold=True),
            labels=dict(left='|w|', bottom='Episode'),
        )

        try:
            pplot = policy.plot
        except AttributeError:
            policy_plot = False
        else:
            pplot(domain, args_ns.episodes)
            policy_plot = True

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
        print(f'Starting run {i+1} / {args_ns.runs};  Running {args_ns.episodes} episodes... (with seed {seed})')
        rnd.seed(seed)

        idx_offset = i * nepisodes
        idx_returns = 0
        idx_gnorms = 0

        returns_run = np.empty(nepisodes)

        policy.reset()
        for e in range(nepisodes):
            algo.episode(env, policy, nsteps=nsteps)

            if args_ns.graph:
                # NOTE policy plot
                if i == 0 and policy_plot is not None:
                    policy.plot_update()

                # TODO plot gradients!
                # # NOTE gradient norm
                # if gradient.dtype == object:
                #     gnorm = np.sqrt(sum(_.sum() for _ in gradient ** 2))
                # else:
                #     gnorm = np.sqrt(np.sum(gradient ** 2))
                # q_gnorms.put((idx_offset + idx_gnorms, gnorm))
                # idx_gnorms += 1

                # NOTE collect returns
                q_returns.put((idx_offset + idx_returns, env.g))
            returns_run[idx_returns] = env.g
            idx_returns += 1

        return returns_run


    if args_ns.processes > 1:  # parallel
        with mp.Pool(processes=args_ns.processes) as pool:
            rets = pool.map(run, range(nruns))
    else:  # sequential
        rets = [run(ri) for ri in range(nruns)]
    rets = np.array(rets)


    try:
        np.save(args_ns.out, rets)
    except AttributeError:
        pass


    if args_ns.graph:
        q_returns.put(None)
        q_gnorms.put(None)

        #  keeps graphics alive
        import IPython
        IPython.embed()

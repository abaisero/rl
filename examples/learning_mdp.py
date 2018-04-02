#!/usr/bin/env python

import logging.config
from logconfig import LOGGING

import rl.mdp as mdp

import rl.mdp.objectives as objectives
import rl.mdp.domains as domains
import rl.mdp.policies as policies
import rl.mdp.algos as algos
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
    parser.add_argument('mdp', type=str, help='MDP name')


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

    domain = mdp.domains.from_fname(args_ns.mdp)
    objective = getattr(mdp.objectives, obj)(domain)


    algo_ns.stepsize = optim.StepSize(.1)  # TODO set using argv

    # TODO joint factory?!
    algo = algos.factory(algo_ns)
    policy = policies.factory(domain, policy_ns)

    print('MDP:', domain)
    print('Algo:', algo)
    print('Policy:', policy)

    print(mdp.env)
    env = mdp.env.Environment(domain, objective)


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

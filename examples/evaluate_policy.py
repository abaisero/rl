#!/usr/bin/env python

import argparse

import logging.config
from logconfig import LOGGING

import multiprocessing as mp
from tqdm import tqdm

import rl.pomdp as pomdp
import rl.pomdp.policies as policies

import numpy as np


if __name__ == '__main__':
    # logging configuration
    logging.config.dictConfig(LOGGING)

    parser = argparse.ArgumentParser(description='Policy Empirical Evaluation')

    parser.add_argument(
        '--out', type=str, default=None, help='output file name')
    parser.add_argument(
        '--processes', type=int, default=mp.cpu_count() - 1,
        help='number of processes')
    parser.add_argument(
        '--gtype', type=str, choices=['longterm', 'discounted'],
        default='longterm', help='return type')
    parser.add_argument(
        '--episodes', type=int, default=1000,
        help='number of training episodes')
    parser.add_argument(
        '--steps', type=int, default=100,
        help='number of steps in an episode')

    parser.add_argument('pomdp', type=str, help='POMDP name')
    parser.add_argument('policy', type=str, help='Policy arguments')

    args = parser.parse_args()
    print(f'Argument Namespace: {args}')

    env = pomdp.Environment.from_fname(args.pomdp)
    policy = policies.factory(env, args.policy)
    print(env)
    print(policy)

    nepisodes = args.episodes
    nsteps = args.steps

    def initializer():
        np.random.seed()

    def episode(i):
        return env.episode(policy, nsteps, gtype=args.gtype)

    rets = np.empty(nepisodes)
    if args.processes > 1:  # parallel
        with mp.Pool(processes=args.processes, initializer=initializer) \
                as pool, tqdm(total=nepisodes) as pbar:
            for i, g in \
                    enumerate(pool.imap_unordered(episode, range(nepisodes))):
                rets.itemset(i, g)
                pbar.update()
    else:  # sequential
        for i in tqdm(range(nepisodes)):
            rets.itemset(i, episode(i))

    try:
        np.save(args.out, rets)
    except AttributeError:
        pass

    import matplotlib.pyplot as plt
    n, _, _ = plt.hist(rets.ravel(), bins='auto', density=True)
    nmax = n.max()

    percentiles = np.percentile(rets, [0, 25, 50, 75, 100])
    plt.vlines(percentiles[0], 0, nmax, colors='red')
    plt.vlines(percentiles[1], 0, nmax, colors='black', linestyles='dashed')
    plt.vlines(percentiles[2], 0, nmax, colors='black')
    plt.vlines(percentiles[3], 0, nmax, colors='black', linestyles='dashed')
    plt.vlines(percentiles[4], 0, nmax, colors='green')
    plt.show()

#!/usr/bin/env python

import sys
import logging
import logging.config
import argparse

import multiprocessing as mp
from tqdm import tqdm, trange

import rl.pomdp as pomdp
import rl.pomdp.policies as policies

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Policy Empirical Evaluation')

    parser.add_argument('--log', metavar='L', type=str, default='info',
                        help='log level')

    parser.add_argument('--out', metavar='F', type=str, default=None,
                        help='output file name')

    parser.add_argument('--processes', metavar='P', type=int,
                        default=mp.cpu_count() - 1, help='number of processes')
    parser.add_argument('--episodes', metavar='E', type=int, default=1000,
                        help='number of episodes in run')
    parser.add_argument('--steps', metavar='S', type=int, default=100,
                        help='number of steps in episode')

    parser.add_argument('--gtype', type=str,
                        choices=['longterm', 'discounted'], default='longterm',
                        help='return type')

    parser.add_argument('env', type=str, help='environment')
    parser.add_argument('policy', type=str, help='policy')

    config = parser.parse_args()
    print(f'Argument Namespace: {config}')

    try:
        level = getattr(logging, config.log.upper())
    except AttributeError:
        level = None

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,

        'formatters': {
            'simple': {
                'format': '{levelname} {name} {message}',
                'style': '{',
            },
        },

        'handlers': {
            'file': {
                'level': level,
                'class': 'logging.FileHandler',
                'filename': 'evaluate.log',
                'mode': 'w',
                'formatter': 'simple',
            },
        },

        'loggers': {
            'rl': {
                'handlers': ['file'],
                'level': 'DEBUG',
                'propagate': True,
            },
        },
    })

    logger = logging.getLogger('rl')
    logger.info(f'Running evaluate.py')
    logger.info(f' - args:  {sys.argv[1:]}')
    logger.info(f' - config:  {config}')

    env = pomdp.Environment.from_fname(config.env)
    policy = policies.factory(env, config.policy)
    print(env)
    print(policy)

    nepisodes = config.episodes
    nsteps = config.steps

    def initializer():
        np.random.seed()

    def episode(i):
        return env.episode(policy, nsteps, gtype=config.gtype)

    returns = np.empty(nepisodes)
    if config.processes > 1:  # parallel
        with mp.Pool(processes=config.processes, initializer=initializer) \
                as pool, tqdm(total=nepisodes) as pbar:
            for i, g in \
                    enumerate(pool.imap_unordered(episode, range(nepisodes))):
                returns.itemset(i, g)
                pbar.update()
    else:  # sequential
        for i in trange(nepisodes):
            returns.itemset(i, episode(i))

    if config.out:
        # save results
        np.save(config.out, returns)

    import matplotlib.pyplot as plt
    n, _, _ = plt.hist(returns.ravel(), bins='auto', density=True)
    nmax = n.max()

    percentiles = np.percentile(returns, [0, 25, 50, 75, 100])
    plt.vlines(percentiles[0], 0, nmax, colors='red')
    plt.vlines(percentiles[1], 0, nmax, colors='black', linestyles='dashed')
    plt.vlines(percentiles[2], 0, nmax, colors='black')
    plt.vlines(percentiles[3], 0, nmax, colors='black', linestyles='dashed')
    plt.vlines(percentiles[4], 0, nmax, colors='green')
    plt.show()

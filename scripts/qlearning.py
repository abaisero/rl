#!/usr/bin/env python

import sys

# import logging.config
# from logconfig import LOGGING

import rl.data as data
import rl.pomdp as pomdp
import rl.pomdp.policies as policies
import rl.pomdp.algos as algos
import rl.optim as optim
import rl.graph as graph
import pyqtgraph as pg

import numpy as np

import multiprocessing as mp
from tqdm import tqdm

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-learning')

    parser.add_argument('--pbar', action='store_true', help='progress bars')
    parser.add_argument('--graph', action='store_true', help='graphics')

    parser.add_argument('--out', metavar='F', type=str, default=None,
                        help='output file name')

    parser.add_argument('--processes', metavar='P', type=int,
                        default=mp.cpu_count() - 1, help='number of processes')
    parser.add_argument('--samples', metavar='S', type=int,
                        default=1, help='number of MC samples')

    parser.add_argument('--runs', metavar='R', type=int, default=10,
                        help='number of learning runs')
    parser.add_argument('--episodes', metavar='E', type=int, default=1000,
                        help='number of episodes in run')
    parser.add_argument('--steps', metavar='S', type=int, default=100,
                        help='number of steps in episode')

    parser.add_argument('env', type=str, help='environment')
    parser.add_argument('policy', type=str, help='policy')

    config = parser.parse_args()
    print(f'Argument Namespace: {config}')

    nprocesses = config.processes
    nsamples = config.samples
    nruns = config.runs
    nepisodes = config.episodes
    nsteps = config.steps

    if config.adam:
        optimizer = optim.Adam()
    else:
        optimizer = optim.GDescent(config.stepsize, config.clip)

    env = pomdp.Environment.from_fname(config.env)
    policy = policies.factory(env, config.policy)

    params = policy.new_params()

    econtext = env.new_context()
    pcontext = policy.new_context(params)
    while True:
        a = policy.sample_a(params, pcontext)
        feedback, econtext1 = env.step(econtext, a)
        pcontext1 = policy.step(params, pcontext, feedback)

        # TODO train policy here...

        econtext = econtext1
        pcontext = pcontext1

#!/usr/bin/env python

import argparse

import logging.config
from logconfig import LOGGING

import rl.pomdp as pomdp

import numpy as np


if __name__ == '__main__':
    # logging configuration
    logging.config.dictConfig(LOGGING)

    parser = argparse.ArgumentParser(description='Human Empirical Evaluation')

    parser.add_argument('--sr', action='store_true', help='Obfuscate rewards.')
    parser.add_argument(
        '--gtype', type=str, choices=['longterm', 'discounted'],
        default='longterm', help='return type')

    parser.add_argument('pomdp', type=str, help='POMDP name')

    args = parser.parse_args()
    print(f'Argument Namespace: {args}')

    env = pomdp.Environment.from_fname(args.pomdp)
    print(env)

    econtext = env.new_context(gtype=args.gtype)

    from random import shuffle

    indices = list(map(str, range(1, env.nactions + 1)))
    print(f'actions: {", ".join(indices)}')
    shuffle(indices)
    adict = dict(zip(indices, env.actions))

    import string
    letters = list(string.ascii_letters[:env.nobs])
    print(f'observations: {", ".join(letters)}')
    shuffle(letters)
    odict = dict(zip(env.obs, letters))

    if args.sr:
        rewards = np.unique(env.model.rewards)
        rewards_ = rewards.copy()
        np.random.shuffle(rewards_)

        def reward_shuffler(r):
            i = np.where(rewards == r)[0].item()
            return rewards_[i]
    else:
        def reward_shuffler(r):
            return r

    # def hidden_r(r):
    #     if r == 10.:
    #         return 1.
    #     elif r == -100.:
    #         return -1.
    #     elif r == -1.:
    #         return 0.

    rs, rlim = [], 10

    while True:
        print()
        print(f't: {econtext.t}')
        ain = input('action: ')

        try:
            a = adict[ain]
        except KeyError:
            print('Action not valid, try again.')
        else:
            feedback, _ = env.step(econtext, a)
            oout = odict[feedback.o]
            rout = reward_shuffler(feedback.r)
            rs.append(rout)
            if len(rs) > rlim:
                rs.pop(0)

            print(f'reward: {rout} {econtext.g} (cum-{rlim}: {sum(rs)})')
            print(f'observation: {oout}')

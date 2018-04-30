from .isgpomdp import isgpomdp
from .entropy import entropy
from .contextful import contextful, contextful2, contextful3
from functools import partial

import argparse


parser = argparse.ArgumentParser(description='Algo')
_subparsers = parser.add_subparsers(title='Algo', dest='algo')
_subparsers.required = True

_parser = _subparsers.add_parser('isgpomdp', help='IsGPOMDP')
_parser.add_argument('--beta', type=float, required=False, default=None)

_parser = _subparsers.add_parser('entropy', help='Entropy')

_parser = _subparsers.add_parser('contextful', help='CONTEXTFUL')
_parser.add_argument('--softmin', type=float, default=None)

_parser = _subparsers.add_parser('contextful2', help='CONTEXTFUL')
_parser.add_argument('--softmin', type=float, default=None)
_parser.add_argument('cfprobs', nargs='+', type=float)

_parser = _subparsers.add_parser('contextful3', help='CONTEXTFUL')
_parser.add_argument('--softmin', type=float, default=None)
_parser.add_argument('cfprobs', nargs='+', type=float)


def factory(argstr):
    args = parser.parse_args(argstr.split())

    # if args.algo == 'gpomdp':
    #     pg = GPOMDP.from_namespace(args)
    #     return PolicyGradient(pg, args.stepsize)
    if args.algo == 'isgpomdp':
        return partial(isgpomdp, beta=args.beta)
    # elif args.algo == 'expgpomdp':
    #     pg = ExpGPOMDP.from_namespace(args)
    #     return PolicyGradient(pg, args.stepsize)
    # elif args.algo == 'cfgpomdp':
    #     pg = CFGPOMDP.from_namespace(args)
    #     return PolicyGradient(pg, args.stepsize)
    elif args.algo == 'entropy':
        return partial(entropy)
    elif args.algo == 'contextful':
        return partial(contextful, l=args.softmin)
    elif args.algo == 'contextful2':
        return partial(contextful2, cfprobs=args.cfprobs, l=args.softmin)
    elif args.algo == 'contextful3':
        return partial(contextful3, cfprobs=args.cfprobs, l=args.softmin)

    raise ValueError(f'Algorithm `{args.algo}` not recognized')

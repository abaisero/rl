# from .softmin import softmin
from .isgpomdp import IsGPOMDP
# from .entropy import entropy
from .contextful import Contextful, Contextful2, Contextful3
from functools import partial

import argparse


parser = argparse.ArgumentParser(description='Algo')
_subparsers = parser.add_subparsers(title='Algo', dest='algo')
_subparsers.required = True

# _parser = _subparsers.add_parser('softmin', help='Softmin')
# _parser.add_argument('l', type=float)
# _parser.add_argument('algo_sm', nargs='+', type=str)

_parser = _subparsers.add_parser('isgpomdp', help='IsGPOMDP')
_parser.add_argument('--beta', type=float, default=None)

_parser = _subparsers.add_parser('entropy', help='Entropy')

_parser = _subparsers.add_parser('contextful', help='CONTEXTFUL')
# _parser.add_argument('--softmin', type=float, default=None)

_parser = _subparsers.add_parser('contextful2', help='CONTEXTFUL2')
# _parser.add_argument('--softmin', type=float, default=None)
_parser.add_argument('cfprobs', nargs='+', type=float)

_parser = _subparsers.add_parser('contextful3', help='CONTEXTFUL3')
# _parser.add_argument('--softmin', type=float, default=None)
_parser.add_argument('--beta', type=float, default=None)
_parser.add_argument('cfprobs', nargs='+', type=float)


def factory(env, policy, argstr):
    args = parser.parse_args(argstr.split())

    # if args.algo == 'softmin':
    #     # TODO how to handle partial?
    #     algo = factory(' '.join(args.algo_sm))
    #     return partial(softmin, algo=algo, l=args.l)

    # if args.algo == 'gpomdp':
    #     pg = GPOMDP.from_namespace(args)
    #     return PolicyGradient(pg, args.stepsize)
    if args.algo == 'isgpomdp':
        beta = env.gamma if args.beta is None else args.beta
        return IsGPOMDP(policy, beta)
        # return partial(isgpomdp, beta=args.beta)
    # elif args.algo == 'expgpomdp':
    #     pg = ExpGPOMDP.from_namespace(args)
    #     return PolicyGradient(pg, args.stepsize)
    # elif args.algo == 'cfgpomdp':
    #     pg = CFGPOMDP.from_namespace(args)
    #     return PolicyGradient(pg, args.stepsize)

    # elif args.algo == 'entropy':
    #     return partial(entropy)
    elif args.algo == 'contextful':
        return Contextful(policy)
    elif args.algo == 'contextful2':
        return Contextful2(policy, args.cfprobs)
        # return partial(contextful2, cfprobs=args.cfprobs)
    elif args.algo == 'contextful3':
        beta = env.gamma if args.beta is None else args.beta
        return Contextful3(policy, beta, args.cfprobs)

    raise ValueError(f'Algorithm `{args.algo}` not recognized')

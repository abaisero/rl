# from .softmin import softmin
from .isgpomdp import IsGPOMDP
from .entropy import Entropy
from .contextful import Contextful, Contextful2, Contextful3
from .entromix import Entromix

import argparse


parser = argparse.ArgumentParser(description='Algo')
_subparsers = parser.add_subparsers(title='Algo', dest='algo')
_subparsers.required = True

_parser = _subparsers.add_parser('isgpomdp', help='IsGPOMDP')
_parser.add_argument('--beta', type=float, default=1.)

_parser = _subparsers.add_parser('entropy', help='Entropy')

_parser = _subparsers.add_parser('contextful', help='CONTEXTFUL')

_parser = _subparsers.add_parser('contextful2', help='CONTEXTFUL2')
_parser.add_argument('cfprobs', nargs='+', type=float)

_parser = _subparsers.add_parser('contextful3', help='CONTEXTFUL3')
_parser.add_argument('cfprobs', nargs='+', type=float)

_parser = _subparsers.add_parser('entromix', help='Entromix')
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

    elif args.algo == 'entropy':
        return Entropy(policy)
    elif args.algo == 'contextful':
        return Contextful(policy)
    elif args.algo == 'contextful2':
        return Contextful2(policy, args.cfprobs)
    elif args.algo == 'contextful3':
        return Contextful3(policy, args.cfprobs)
    elif args.algo == 'entromix':
        return Entromix(policy, args.cfprobs)

    raise ValueError(f'Algorithm `{args.algo}` not recognized')

import argparse

# TODO remove all these;  perhaps don't even have a single hub of stuff...
# There is FSC and then all the rest must be built individually?
from .cf import CF
from .reactive import Reactive
from .fsc import FSC
from .fsc_sparse import FSC_Sparse
from .fsc_structured import FSC_Structured
from .fsc_file import FSC_File
from .fsc_reactive import FSC_Reactive

from .astrat import AStrategy
from .ostrat import OStrategy, OStrategy_Sparse, OStrategy_Reactive
from .nvalue import Value


parser = argparse.ArgumentParser(description='Policy')
_subparsers = parser.add_subparsers(title='Policy', dest='policy')
_subparsers.required = True

_parser = _subparsers.add_parser('cf', help='CF')

_parser = _subparsers.add_parser('reactive', help='Reactive')

_parser = _subparsers.add_parser('fsc', help='FSC')
_parser.add_argument('n', type=int, help='number of nodes')

_parser = _subparsers.add_parser('fsc_sparse', help='FSC_Sparse')
_parser.add_argument('n', type=int, help='number of nodes')
_parser.add_argument('k', type=int, help='sparsity')

_parser = _subparsers.add_parser('fsc_structured', help='FSC_Structured')
_parser.add_argument('fss', type=str, help='finite state structure file')

_parser = _subparsers.add_parser('fsc_file', help='FSC_File')
_parser.add_argument('fsc', type=str, help='finite state control file')

_parser = _subparsers.add_parser('fsc_reactive', help='FSC_Reactive')
_parser.add_argument('k', type=int, help='number of observations')

# _parser = _subparsers.add_parser('qlearning', help='Qlearning')
# _parser.add_argument('n', type=int, help='number of observations')


def factory(env, config, argstr):
    args = parser.parse_args(argstr.split())

    # if getattr(args, 'belief', False):
    #     args.belief = False
    #     fsc = factory(env, args)
    #     return BeliefFSC(env, fsc)

    if args.policy == 'cf':
        raise NotImplementedError
        return CF.from_namespace(env, args)
    elif args.policy == 'reactive':
        ostrat = OStrategy_Reactive(env.nobs, 1)
        astrat = AStrategy(ostrat.nnodes, env.nactions, gain=config.gain)
        critic = Value(ostrat.nnodes) if config.critic else None
        policy = FSC(astrat, ostrat, critic=critic)
    elif args.policy == 'fsc':
        astrat = AStrategy(args.n, env.nactions, gain=config.gain)
        ostrat = OStrategy(args.n, env.nobs, gain=config.gain)
        critic = Value(args.n) if config.critic else None
        policy = FSC(astrat, ostrat, critic=critic)
    elif args.policy == 'fsc_sparse':
        astrat = policies.AStrategy(args.n, env.nactions, gain=config.gain)
        ostrat = policies.OStrategy_Sparse(args.n, env.nobs, args.k, gain=config.gain)
        critic = Value(args.n) if config.critic else None
        policy = FSC(astrat, ostrat, critic=critic)
    elif args.policy == 'fsc_structured':
        raise NotImplementedError
        return FSC_Structured.from_namespace(env, args)
    elif args.policy == 'fsc_file':
        raise NotImplementedError
        return FSC_File.from_namespace(env, args)
    elif args.policy == 'fsc_reactive':
        ostrat = OStrategy_Reactive(env.nobs, args.k)
        astrat = AStrategy(ostrat.nnodes, env.nactions, gain=config.gain)
        critic = Value(ostrat.nnodes) if config.critic else None
        policy = FSC(astrat, ostrat, critic=critic)
    else:
        raise ValueError(f'Policy `{args.policy}` not recognized')

    return policy

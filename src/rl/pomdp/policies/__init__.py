import argparse

from .cf import CF
from .reactive import Reactive
from .fsc import FSC
from .fsc_sparse import FSC_Sparse
from .fsc_structured import FSC_Structured
from .fsc_file import FSC_File


parser = argparse.ArgumentParser(description='Policy')
_subparsers = parser.add_subparsers(title='Policy', dest='policy')
_subparsers.required = True

_parser = _subparsers.add_parser('cf', help='CF')

_parser = _subparsers.add_parser('reactive', help='Reactive')

_parser = _subparsers.add_parser('fsc', help='FSC')
_parser.add_argument('n', type=int)

_parser = _subparsers.add_parser('fsc_sparse', help='FSC_Sparse')
_parser.add_argument('n', type=int)
_parser.add_argument('k', type=int)

_parser = _subparsers.add_parser('fsc_structured', help='FSC_Structured')
_parser.add_argument('fss', type=str)

_parser = _subparsers.add_parser('fsc_file', help='FSC_File')
_parser.add_argument('fsc', type=str)


def factory(env, argstr):
    args = parser.parse_args(argstr.split())

    # if getattr(args, 'belief', False):
    #     args.belief = False
    #     fsc = factory(env, args)
    #     return BeliefFSC(env, fsc)

    if args.policy == 'cf':
        return CF.from_namespace(env, args)
    elif args.policy == 'reactive':
        return Reactive.from_namespace(env, args)
    elif args.policy == 'fsc':
        return FSC.from_namespace(env, args)
    elif args.policy == 'fsc_sparse':
        return FSC_Sparse.from_namespace(env, args)
    elif args.policy == 'fsc_structured':
        return FSC_Structured.from_namespace(env, args)

    if args.policy == 'fsc_file':
        return FSC_File.from_namespace(env, args)

    raise ValueError(f'Policy `{args.policy}` not recognized')

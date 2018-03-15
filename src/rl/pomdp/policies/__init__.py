from .policy import Policy

from .random import Random
from .blind import Blind
from .reactive import Reactive
from .fsc import FSC
from .sparse_fsc import SparseFSC
from .structured_fsc import StructuredFSC


def add_subparsers(parser, title='Policy', dest='policy'):
    subparsers = parser.add_subparsers(title=title, dest=dest)
    subparsers.required = True

    _parser = FSC.parser('fsc')
    subparsers.add_parser('fsc', parents=[_parser], help='FSC')

    _parser = SparseFSC.parser('fsc_sparse')
    subparsers.add_parser('fsc_sparse', parents=[_parser], help='Sparse FSC')

    _parser = StructuredFSC.parser('fsc_structured')
    subparsers.add_parser('fsc_structured', parents=[_parser], help='Structured FSC')


def policy_cls(name):
    # TODO random, blind and reactive...
    if name == 'fsc':
        return FSC
    elif name == 'fsc_sparse':
        return SparseFSC
    elif name == 'fsc_structured':
        return StructuredFSC
    else:
        raise ValueError(f'Policy name {name} not recognized')

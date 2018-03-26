from .policy import Policy

from .random import Random
from .blind import Blind
from .reactive import Reactive
from .fsc import FSC
from .sparse_fsc import SparseFSC
from .structured_fsc import StructuredFSC
from .bfsc import BeliefFSC

from ._argparser import parser





# def add_subparsers(parser, title='Policy', dest='policy'):
#     subparsers = parser.add_subparsers(title=title, dest=dest)

#     _parser = FSC.parser('fsc')
#     p = subparsers.add_parser('fsc', parents=[_parser], help='FSC')

#     print('this')
#     global _blarg
#     if not _blarg:
#         _blarg = True
#         add_subparsers(p)

#     _parser = SparseFSC.parser('fsc_sparse')
#     subparsers.add_parser('fsc_sparse', parents=[_parser], help='Sparse FSC')

#     _parser = StructuredFSC.parser('fsc_structured')
#     subparsers.add_parser('fsc_structured', parents=[_parser], help='Structured FSC')


# def policy_cls(name, ns):
#     # TODO random, blind and reactive...
#     # TODO FUCK FUCK FUCK factory!
#     if name == 'fsc':
#         return FSC
#     elif name == 'fsc_sparse':
#         return SparseFSC
#     elif name == 'fsc_structured':
#         return StructuredFSC
#     else:
#         raise ValueError(f'Policy name {name} not recognized')


def factory(domain, ns):
    try:
        if ns.belief:
            ns.belief = False
            fsc = factory(domain, ns)
            return BeliefFSC(domain, fsc)
    except AttributeError:
        pass

    if ns.policy == 'fsc':
        return FSC.from_namespace(domain, ns)
    elif ns.policy == 'fsc_sparse':
        return SparseFSC.from_namespace(domain, ns)
    elif ns.policy == 'fsc_structured':
        return StructuredFSC.from_namespace(domain, ns)

    raise ValueError(f'Policy `{ns.policy}` not recognized')

from .pgradient import PolicyGradient
from .psearch import PolicySearch

from .gpomdp import GPOMDP
from .isgpomdp import IsGPOMDP
from .expgpomdp import ExpGPOMDP
from .cfgpomdp import CFGPOMDP
from .conjpomdp import CONJPOMDP

from ._argparser import parser





# _parser = ExpGPOMDP.parser('expgpomdp')
# _subparsers.add_parser('expgpomdp', parents=[_parser], help='ExpGPOMDP')


# def add_subparsers(parser, title='Algo', dest='algo'):
#     subparsers = parser.add_subparsers(title=title, dest=dest)
#     subparsers.required = True

#     _parser = GPOMDP.parser('gpomdp')
#     subparsers.add_parser('gpomdp', parents=[_parser], help='GPOMDP')

#     _parser = IsGPOMDP.parser('isgpomdp')
#     subparsers.add_parser('isgpomdp', parents=[_parser], help='IsGPOMDP')

#     _parser = ExpGPOMDP.parser('expgpomdp')
#     subparsers.add_parser('expgpomdp', parents=[_parser], help='ExpGPOMDP')


def factory(ns):
    if ns.algo == 'gpomdp':
        pg = GPOMDP.from_namespace(ns)
        return PolicyGradient(pg, ns.stepsize)
    elif ns.algo == 'isgpomdp':
        pg = IsGPOMDP.from_namespace(ns)
        return PolicyGradient(pg, ns.stepsize)
    elif ns.algo == 'expgpomdp':
        pg = ExpGPOMDP.from_namespace(ns)
        return PolicyGradient(pg, ns.stepsize)
    elif ns.algo == 'cfgpomdp':
        pg = CFGPOMDP.from_namespace(ns)
        return PolicyGradient(pg, ns.stepsize)

    raise ValueError(f'Algorithm `{ns.algo}` not recognized')

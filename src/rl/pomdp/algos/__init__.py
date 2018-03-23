from .pgradient import PolicyGradient
from .psearch import PolicySearch

from .pgrads import GPOMDP, IsGPOMDP, ExpGPOMDP

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


def factory(policy, ns):
    if ns.algo == 'gpomdp':
        pgrad = GPOMDP.from_namespace(policy, ns)
        return PolicyGradient(policy, pgrad, ns.stepsize)
    elif ns.algo == 'isgpomdp':
        pgrad = IsGPOMDP.from_namespace(policy, ns)
        return PolicyGradient(policy, pgrad, ns.stepsize)
    elif ns.algo == 'expgpomdp':
        pgrad = ExpGPOMDP.from_namespace(policy, ns)
        return PolicyGradient(policy, pgrad, ns.stepsize)

    raise ValueError(f'Algorithm `{ns.algo}` not recognized')

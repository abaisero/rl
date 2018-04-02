# from .policy import Policy
# from .softmax import Softmax

# from .random import Random
# from .egreedy import eGreedy
# from .qsoftmax import QSoftmax
# from .psoftmax import PSoftmax

from .policy import Policy

from .softmax import Softmax

from ._argparser import parser


def factory(domain, ns):
    if ns.policy == 'softmax':
        return Softmax.from_namespace(domain, ns)

    raise ValueError(f'Policy `{ns.policy}` not recognized')

# from .agent import Agent, PreAgent
# from .montecarlo import MonteCarloES, MonteCarloControl
# from .sarsa import SARSA, SARSA_l, ExpectedSARSA
# # from .qlearning import Qlearning, Qlearning_l, DoubleQlearning
# from .qlearning import Qlearning, Qlearning_l
# from .td import TD, TD_l

from .pgradient import PolicyGradient

from .reinforce import REINFORCE

from ._argparser import parser


def factory(ns):
    if ns.algo == 'reinforce':
        pg = REINFORCE.from_namespace(ns)
        return PolicyGradient(pg, ns.stepsize)

    raise ValueError(f'Algorithm `{ns.algo}` not recognized')

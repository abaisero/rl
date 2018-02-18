class Algo(object):
    def __init__(self, sys, model, policy, Q):
        self.sys = sys
        self.model = model
        self.policy = policy
        self.Q = Q

    def run(self, s, verbose=False):
        raise NotImplementedError


from .mc import MC
from .mcts import MCTS
# from .pomcp import POMCP

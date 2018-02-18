class Algo(object):
    def __init__(self, sys, model, policy, Q):
        self.sys = sys
        self.model = model
        self.policy = policy
        self.Q = Q

    def run(self, s, verbose=False):
        raise NotImplementedError


from .sarsa import SARSA, SARSA_l
from .qlearning import Qlearning, Qlearning_l

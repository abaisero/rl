import numpy as np
from scipy.stats import multivariate_normal

from rl.problems import tstate, State, Action, SAPair, Model
from rl.problems.mdp import MDP
from rl.values import ActionValues_Tabular, ActionValues_Linear, ActionValues_LinearBayesian
from rl.policy import Policy_random, Policy_egreedy, Policy_UCB, UCB_confidence_Q
from rl.algo.search import TDSearch

from pytk.util import true_every


class GolfState(State):
    discrete = True

    def __init__(self, dist, nstrokes):
        self.dist = dist
        self.nstrokes = nstrokes

        degree = 3
        self.phi = np.empty(degree + 2)
        self.phi[0] = nstrokes
        self.phi[1:] = np.vander([dist], degree + 1)

    def __hash__(self):
        return hash((self.dist, self.nstrokes))

    def __eq__(self, other):
        try:
            return self.dist == other.dist and self.nstrokes == other.nstrokes
        except AttributeError:
            return False

    def __str__(self):
        return 'S(dist={}, nstrokes={})'.format(self.dist, self.nstrokes)


class GolfAction(Action):
    discrete = True

    def __init__(self, club, clubs):
        self.club = club
        self.clubs = clubs

        # TODO fix this
        # self.phi = (clubs == club).astype(np.float64)
        self.phi = (clubs == club).astype(np.int64)

    def __hash__(self):
        return hash(self.club)

    def __eq__(self, other):
        return self.club == other.club

    def __str__(self):
        return 'A(club={})'.format(self.club)


class GolfModel(Model):
    def __init__(self, s0_dists):
        super(GolfModel, self).__init__()
        self.s0_dists = s0_dists

    def sample_s0(self):
        dist = rnd.choice(self.s0_dists)
        return GolfState(dist, 0)

    def sample_s1(self, s0, a):
        s1dist = abs(s0.dist - a.club)
        if s1dist == 0:
            return tstate
        return GolfState(s1dist, s0.nstrokes + 1)

    def sample_r(self, s0, a, s1):
        return 0. if s1 == tstate else -1.


class GolfMDP(MDP):
    """ Golf game MDP """
    def __init__(self, s0_dists, clubs):
        super(GolfMDP, self).__init__(GolfModel(s0_dists))

        self.clubs = clubs
        self.maxr = 1

        # self.statelist_start =   # TODO I don't think I need this?
        self.actionlist = [GolfAction(club, clubs) for club in clubs]


if __name__ == '__main__':
    import numpy.random as rnd
    rnd.seed(0)

    # NOTE this still breaks things a bit
    s0_from, s0_to = 90, 110
    # s0_from, s0_to = 15, 25
    # s0_from, s0_to = 19, 21
    s0_dists = np.arange(s0_from, s0_to+1)
    # clubs = np.arange(1, 11)
    clubs = np.array([1, 2, 5, 10])

    mdp = GolfMDP(s0_dists, clubs)

    # NOTE choose between the following models

    # model 1: tabular AV, UCB policy
    # works with both qlearning and sarsa
    # Q = ActionValues_Tabular()
    # def Q_confidence(sa): return UCB_confidence_Q(sa, Q)
    # policy = Policy_UCB(Q.value, Q_confidence, beta=mdp.maxr)

    # model 2: linear AV, egreedy policy
    # doesn't work.  Best guess:  exploration does not cancel out bad updates
    # Q = ActionValues_Linear(.1)  # TODO I think right now the alpha is not used
    # policy = Policy_egreedy(Q, .1)

    # model 3: bayesian linear AV, UCB policy
    # works with both qlearning and sarsa (takes longer)
    Q = ActionValues_LinearBayesian(l2=100, s2=1)
    policy = Policy_UCB(Q.value, Q.confidence, beta=mdp.maxr)

    # NOTE choose update method
    # Q.update_method = 'sarsa'
    Q.update_method = 'qlearning'

    # TODO how to parametrize update function?  How to select here whether to use qlearning, or sarsa?

    tds = TDSearch(mdp, mdp.model, policy, Q)

    for i in range(100):
        s0 = mdp.model.sample_s0()
        tds.run(s0, 1000, verbose=true_every(100))

    # TODO code to evaluate current solution:
    #  * print V(s0)
    #  * simulate env many times with greedy policy, MC-evaluate V(s0)
    # * compare the two.

    # Also print out a few simulations to see if something weird happens

    # TODO update UCB somehow because fapprox does not have update counts

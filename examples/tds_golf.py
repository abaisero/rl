import numpy as np
from scipy.stats import norm

from rl.problems import State, Action, SAPair, Model
from rl.problems.mdp import MDP
from rl.values import ActionValues_Tabular, ActionValues_Linear, ActionValues_LinearBayesian
from rl.policy import Policy_random, Policy_egreedy, Policy_UCB
from rl.algo.search import TDSearch

from pytk.util import true_every


class GolfState(State):
    discrete = False

    def __init__(self, dist, nstrokes):
        self.dist = dist
        self.nstrokes = nstrokes
        self.terminal = dist == 0

        degree = 3
        self.phi = np.empty(degree + 2)
        self.phi[0] = nstrokes
        self.phi[1:] = np.vander([dist], degree + 1)

        degree = 3
        self.phi = np.vander([dist], degree + 1).ravel()

    # TODO I don't really need this when using value function approx, right?!
    def __hash__(self):
        return hash((self.dist, self.nstrokes))

    def __eq__(self, other):
        try:
            return self.dist == other.dist and self.nstrokes == other.nstrokes
        except AttributeError:
            return False

    def __str__(self):
        return 'S(dist={:2.1f}, nstrokes={})'.format(self.dist, self.nstrokes)


class GolfAction(Action):
    discrete = True

    def __init__(self, club, nclub, nclubs):
        self.club = club
        # self.nclub = nclub
        # self.nclubs = nclubs

        # TODO fix this need better wait to define clubs
        self.phi = np.eye(nclubs)[nclub]

    def __hash__(self):
        return hash(self.club)

    def __eq__(self, other):
        return self.club == other.club

    def __str__(self):
        return 'A(club={})'.format(self.club)


class GolfModel(Model):
    def __init__(self, s0dist_m, s0dist_s):
        super(GolfModel, self).__init__()
        self.s0dist_rv = norm(s0dist_m, s0dist_s)

        self.maxr = 10

    def sample_s0(self):
        s0dist = abs(self.s0dist_rv.rvs())
        return GolfState(s0dist, 0)

    def sample_s1(self, s0, a):
        # TODO choose action representation

        # if s0.nstrokes == 100:
        #     return self.env.terminal

        # cl, st = a

        # if cl == 'wood':
        #     m, s, p = 100, 5, 1
        # elif cl == 'iron':
        #     m, s, p = 10, .5, 2
        # elif cl == 'putter':
        #     m, s, p = 1, .05, 10

        # x = st * multivariate_normal.rvs(m, s ** 2)
        # if s0.dist <= x < p * s0.dist:
        #     return self.env.terminal

        # s1dist = abs(s0.dist - x)
        # return GolfState(s1dist, s0.nstrokes + 1)

        n, m, s, p = a.club

        x = norm.rvs(m, s)
        if p * x <= s0.dist <= x:
            s1dist = 0.
        else:
            s1dist = abs(s0.dist - x)
        return GolfState(s1dist, s0.nstrokes + 1)

    def sample_r(self, s0, a, s1):
        return 10. if s1.terminal else -1.


class GolfMDP(MDP):
    """ Golf game MDP """
    def __init__(self, s0dist_m, s0dist_s, clubs):
        super(GolfMDP, self).__init__(GolfModel(s0dist_m, s0dist_s))

        self.clubs = clubs

        # self.clubs = ['wood', 'iron', 'putter']
        # self.strengths = [.5, 1, 2]

        nclubs = len(clubs)
        self.actionlist = [GolfAction(club, ci, nclubs) for ci, club in enumerate(clubs)]


if __name__ == '__main__':
    import numpy.random as rnd
    rnd.seed(0)

    # env = GolfEnv()
    # # Q = ActionValues_Linear(env, l=.1, a=.1)
    # Q = ActionValues_LinearBayesian(env, l=100., s2=10.)
    # policy = Policy_UCB(env.actions, Q.value, Q.confidence, beta=10.)
    # # policy = Policy_egreedy(env.actions, Q, .5)

    # s0dist_m, s0dist_s = 20, 5
    s0dist_m, s0dist_s = 50, 0
    clubs = [
        ('wood', 10, 5, .9),
        ('iron', 5, .5, .5),
        ('putt', 1, .05, 0),
    ]

    mdp = GolfMDP(s0dist_m, s0dist_s, clubs)

    # NOTE linear AV, egreedy policy
    # TODO this one is not working
    # Q = ActionValues_Linear(.1)  # TODO I think right now the alpha is not used
    # policy = Policy_egreedy(Q, .1)

    # NOTE bayesian linear AV, UCB policy
    Q = ActionValues_LinearBayesian(l2=100, s2=1)
    policy = Policy_UCB(Q.value, Q.confidence, beta=mdp.model.maxr)

    Q.update_method = 'sarsa'
    # Q.update_method = 'qlearning'

    tds = TDSearch(mdp, mdp.model, policy, Q)

    from rl.algo.greedy import GreedyAgent
    ga = GreedyAgent(mdp, mdp.model, Q)

    # v = true_every(100)
    for i in xrange(10000):
        s0 = mdp.model.sample_s0()
        tds.run(s0, 1)


        if i % 100 == 0:
            s0 = mdp.model.sample_s0()
            m, v = ga.evaluate(s0, 1000)

            print '---'
            print 'Empirical: {:>6.2f}; {:>6.2f}'.format(m, v)

        #     actions = mdp.actions(s0)
        #     a = Q.optim_action(actions, s0)
        #     sa = SAPair(s0, a)
        #     Qm = Q.value(sa)
        #     Qv = Q.confidence(sa)

        #     print 'Model    : {:>6.2f}; {:>6.2f}'.format(Qm, Qv)


    # TODO code to evaluate current solution:
    #  * print V(s0)
    #  * simulate env many times with greedy policy, MC-evaluate V(s0)
    # * compare the two.

    # Also print out a few simulations to see if something weird happens

    # TODO update UCB somehow because fapprox does not have update counts

import numpy as np
from scipy.stats import norm

from rl.problems import State, Action, SAPair, Dynamics, Task, Model
from rl.problems.mdp import MDP
from rl.values import Values_Tabular, Values_Linear, Values_LinearBayesian
from rl.policy import Policy_random, Policy_egreedy, Policy_UCB
from rl.algo.td import SARSA, SARSA_l, Qlearning, Qlearning_l

from pytk.util import true_every


class GolfState(State):
    discrete = False

    def __init__(self, dist, nstrokes):
        self.setkey((dist, nstrokes))

        self.dist = dist
        self.nstrokes = nstrokes
        self.terminal = dist == 0

        # NOTE state value does not depend on nstrokes
        # degree = 3
        # self.phi = np.empty(degree + 2)
        # self.phi[0] = nstrokes
        # self.phi[1:] = np.vander([dist], degree + 1)

        degree = 3
        self.phi = np.vander([dist], degree + 1).ravel()

    def __str__(self):
        return 'S(dist={:2.1f}, nstrokes={})'.format(self.dist, self.nstrokes)


class GolfAction(Action):
    discrete = True

    def __init__(self, club, nclub, nclubs):
        self.setkey((club,))

        self.club = club

        # TODO fix this need better wait to define clubs
        self.phi = np.eye(nclubs)[nclub]

    def __str__(self):
        return 'A(club={})'.format(self.club)


class GolfDynamics(Dynamics):
    def __init__(self, s0dist_m, s0dist_s):
        super(GolfDynamics, self).__init__()
        self.s0dist_rv = norm(s0dist_m, s0dist_s)

    def sample_s0(self):
        s0dist = abs(self.s0dist_rv.rvs())
        return GolfState(s0dist, 0)

    def sample_s1(self, s0, a):
        n, m, s, p = a.club

        x = norm.rvs(m, s)
        if p * x <= s0.dist <= x:
            s1dist = 0.
        else:
            s1dist = abs(s0.dist - x)
        return GolfState(s1dist, s0.nstrokes + 1)


class GolfTask(Task):
    def __init__(self, hole_r):
        super(GolfTask, self).__init__()
        self.hole_r = hole_r
        self.maxr = max(1, hole_r)

    def sample_r(self, s0, a, s1):
        return self.hole_r if s1.terminal else -1.


class GolfModel(Model):
    def __init__(self, s0dist_m, s0dist_s):
        super(GolfModel, self).__init__()
        self.s0dist_rv = norm(s0dist_m, s0dist_s)

        self.maxr = 10

    def sample_s0(self):
        s0dist = abs(self.s0dist_rv.rvs())
        return GolfState(s0dist, 0)

    def sample_s1(self, s0, a):
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
    def __init__(self, s0dist_m, s0dist_s, hole_r, clubs):
        dyna = GolfDynamics(s0dist_m, s0dist_s)
        task = GolfTask(hole_r)
        super(GolfMDP, self).__init__(Model(dyna, task))

        self.clubs = clubs

        # self.clubs = ['wood', 'iron', 'putter']
        # self.strengths = [.5, 1, 2]

        nclubs = len(clubs)
        self.actionlist = [GolfAction(club, ci, nclubs) for ci, club in enumerate(clubs)]


if __name__ == '__main__':
    import numpy.random as rnd
    rnd.seed(0)

    # s0dist_m, s0dist_s = 20, 5
    s0dist_m, s0dist_s = 50, 0
    clubs = [
        ('wood', 10, 5, .9),
        ('iron', 5, .5, .5),
        ('putt', 1, .05, 0),
    ]

    mdp = GolfMDP(s0dist_m, s0dist_s, 10., clubs)

    # NOTE linear AV, egreedy policy
    # TODO doesn't work
    # Q = Values_Linear.Q(.1)
    # policy = Policy_egreedy.Q(Q, .1)

    # NOTE bayesian linear AV, UCB policy
    Q = Values_LinearBayesian.Q(l2=100, s2=1)
    policy = Policy_UCB.Q(Q.value, Q.confidence, beta=mdp.model.task.maxr)

    # NOTE algorithm
    algo = SARSA(mdp, mdp.model, policy, Q)  # Equivalent to SARSA_l(0.)
    # algo = SARSA_l(mdp, mdp.model, policy, Q, .5)
    # algo = Qlearning(mdp, mdp.model, policy, Q)

    # TODO not implemented yet
    # algo = MC(mdp, mdp.model, policy, Q)  # Equivalent to SARSA_l(1.)
    # algo = Qlearning_l(mdp, mdp.model, policy, Q, .5)


    nepisodes = 20000

    verbose = true_every(100)
    for i in xrange(nepisodes):
        s0 = mdp.model.dynamics.sample_s0()
        algo.run(s0, verbose=verbose.true)


    #     if i % 100 == 0:
    #         s0 = mdp.model.sample_s0()
    #         m, v = ga.evaluate(s0, 1000)

    #         print '---'
    #         print 'Empirical: {:>6.2f}; {:>6.2f}'.format(m, v)

    #     #     actions = mdp.actions(s0)
    #     #     a = Q.optim_action(actions, s0)
    #     #     sa = SAPair(s0, a)
    #     #     Qm = Q.value(sa)
    #     #     Qv = Q.confidence(sa)

    #     #     print 'Model    : {:>6.2f}; {:>6.2f}'.format(Qm, Qv)


    # # TODO code to evaluate current solution:
    # #  * print V(s0)
    # #  * simulate env many times with greedy policy, MC-evaluate V(s0)
    # # * compare the two.

    # # Also print out a few simulations to see if something weird happens

    # # TODO update UCB somehow because fapprox does not have update counts

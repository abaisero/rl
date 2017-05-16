import numpy as np
import numpy.random as rnd

from pytk.util import Keyable
from pytk.more_collections import defaultdict_noinsert


taction = type(
    'TerminalAction',
    (object,),
    dict(__str__=lambda self: 'A(Terminal)'),
)()


class State(Keyable):
    pass


class Action(Keyable):
    pass


class SAPairException(Exception):
    pass


class SAPair(Keyable):
    def __init__(self, s=None, a=None):
        if s is None and a is None:
            raise SAPairException('Either s or a has to be given')
        self.setkey((s, a))

        self.s = s
        self.a = a

    @property
    def phi(self):
        if self.s is None:
            return self.a.phi
        elif self.a is None:
            return self.s.phi

        if not self.a.discrete:
            raise SAPairException()

        return np.outer(self.a.phi, self.s.phi).ravel()

    def __str__(self):
        return 'SA({}, {})'.format(self.s, self.a)

    def __repr__(self):
        return str(self)


# def phi_sa(s=None, a=None):
#     if s is None and a is None:
#         raise SAPairException('Either s or a has to be given')

#     if self.s is None:
#         return self.a.phi
#     elif self.a is None:
#         return self.s.phi

#     if not self.a.discrete:
#         raise SAPairException()

#     return np.outer(self.s.phi, self.a.phi).ravel()
































class Dynamics(object):
    def dist_s0(self):
        raise NotImplementedError

    def dist_s1(self, s0, a):
        raise NotImplementedError

    def pr_s0(self, s0):
        return self.dist_s0()[s0]

    def pr_s1(self, s0, a, s1):
        return self.dist_s1(s0, a)[s1]

    def sample_s0(self):
        dist_s0 = self.dist_s0()
        prs_s0 = [pr_s0 for _, pr_s0 in dist_s0.viewitems()]
        s0i = rnd.choice(len(dist_s0), p=prs_s0)
        s0 = dist_s0[s0i][0]
        return s0

    def sample_s1(self, s0, a):
        dist_s1 = self.dist_s1(s0, a)
        prs_s1 = [pr_s1 for _, pr_s1 in dist_s1.viewitems()]
        s1i = rnd.choice(len(dist_s1), p=prs_s1)
        s1 = dist_s1[s1i][0]
        return s1


class Task(object):
    def __init__(self, gamma=None):
        if gamma is None: gamma = 1.
        self.gamma = gamma

    def dist_r(self, s0, a, s1):
        raise NotImplementedError

    def pr_r(self, s0, a, s1, r):
        return self.dist_r(s0, a, s1)[r]

    def sample_r(self, s0, a, s1):
        dist_r = self.dist_r(s0, a, s1).items()
        prs_r = [pr_r for _, pr_r in dist_r]
        ri = rnd.choice(len(dist_r), p=prs_r)
        r = dist_r[ri][0]
        return r

    def E_r(self, s0, a, s1):
        return sum(r * pr_r for r, pr_r in self.dist_r(s0, a, s1).viewitems())


class Model(object):
    def __init__(self, dynamics, task):
        self.dynamics = dynamics
        self.task = task

    def T(self, s0, a, s1):
        return self.dynamics.pr_s1(s0, a, s1)

    def R(self, s0, a, s1):
        return self.task.E_r(s0, a, s1)

    def dist_rs1(self, s0, a):
        dist_rs1 = defaultdict_noinsert(float)  # float is equivalent to lambda: 0.
        for s1, pr_s1 in self.dynamics.dist_s1(s0, a).viewitems():
            for r, pr_r in self.task.dist_r(s0, a, s1).viewitems():
                dist_rs1[r, s1] = pr_r * pr_s1
        return dist_rs1

    def pr_rs1(self, s0, a, r, s1):
        pr_s1 = self.dynamics.pr_s1(s0, a, s1)
        pr_r = self.task.pr_r(s0, a, s1, r)
        return pr_s1 * pr_r

    def sample_rs1(self, s0, a):
        s1 = self.dynamics.sample_s1(s0, a)
        r = self.task.sample_r(s0, a, s1)
        return r, s1





# case 1 subclass the model structures
# case 2 have a generic struct and add them dynamically
# case 3 they don't really have a state, just feed them a function... except that in case 2 there is a state


# # NOTE probably not very useful for now;  maybe in the future?
# class Task_online(Task):
#     def __init__(self, gamma=None):
#         super(Task_online, self).__init__(gamma)
#         self.__dist_r = defaultdict_noinsert(float)  # float is equivalent to lambda: 0.

#     def add_pr_r(self, s0, a, s1, r, pr_r):
#         self.__dist_r[s0, a, s1, r] += pr_r

#     def dist_r(self, s0, a, s1):
#         dist_r = defaultdict_noinsert(float)  # float is equivalent to lambda: 0.
#         for (s0, a, s1, r), pr_r in self.__dist_r.viewitems():
#             dist_r[r] = pr_r
#         return dist_r


# # NOTE probably not very sueful for now;  maybe redesign in the future
# class Task_decorated(Task):
#     def set_dist_r(self, dist_r):
#         self.dist_r = types.MethodType(dist_r, self)
#         return dist_r

# task = Task_decorated()

# @task.set_dist_r
# def dist_r(self, ):


class System(object):
    """ System defines a statespace, actionspace, and dynamics """
    def __init__(self, model):
        self.model = model

    def states(self, wterm=False):
        return (s for s in self.statelist if wterm or not s.terminal)

    def actions(self, s):
        """ if the actionset depends on the state, this should be overridden """
        if s.terminal:
            return [taction]
        return self.actionlist


class RLProblem(object):
    """ RLProblem defines statespace, actionspace, and dynamics """

    def __init__(self, model):
        self.model = model

    def actions(self, s):
        """ if the actionset depends on the state, this should be overridden """
        if s.terminal:
            return [taction]
        return self.actionlist

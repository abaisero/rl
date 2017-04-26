from __future__ import division

import math

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from GPy.models import GPRegression
from GPy.kern import Linear
from rl.learning import LearningRate_geom, LearningRate_const

from rl.problems import tstate, taction, SAPair


class ValuesException(Exception):
    pass


class Values(object):
    def value(self, sa):
        raise NotImplementedError

    def confidence(self, sa):
        raise NotImplementedError

    def update_target(self, target, sa):
        """ target is either the actual target, or just the instant reward """
        raise NotImplementedError

    def __call__(self, sa):
        return self.value(sa)


class Values_Tabular(Values):
    def __init__(self, initv=0.):
        super(Values_Tabular, self).__init__()
        self.initv = initv
        self.sadict = {}
        self.sdict = {}

    def vn(self, sa):
        return self.sadict.get(sa, (self.initv, 0))

    def value(self, sa):
        if sa.s is tstate:
            return 0.
        return self.vn(sa)[0]

    def nupdates(self, sa):
        return self.vn(sa)[1]

    def nupdates_s(self, sa):
        return self.sdict.get(sa, 0)

    def update_target(self, target, sa):
        v = self.value(sa)
        n = self.nupdates(sa)

        v += (target - v) / (n + 1)
        n += 1

        self.sadict[sa] = v, n
        self.sdict[sa.s] = self.nupdates_s(sa.s) + 1


class Values_Linear(Values):
    def __init__(self, a=1.):
        super(Values_Linear, self).__init__()
        self.a = a

        self.beta = None

    def value(self, sa):
        if sa.s is tstate:
            return 0.
        if self.beta is None:
            return 0.
        return np.dot(sa.phi, self.beta)

    def update_target(self, target, sa):
        if self.beta is None:
            ndim = len(sa.phi)
            self.beta = np.zeros(ndim)
        # TODO is this right?
        # dbeta = self.a * (target - self.value(sa)) * sa.phi
        try:
            self.n += 1
        except:
            self.n = 1

        dbeta = (target - self.value(sa)) * sa.phi / np.dot(sa.phi, sa.phi) / self.n
        # dbeta = (target - self.value(sa)) * sa.phi / np.dot(sa.phi, sa.phi)
        # dbeta = (target - self.value(sa)) * sa.phi / self.n
        # print '==='
        # print target, self.value(sa), target - self.value(sa)
        # print sa.phi
        # print dbeta
        self.beta += dbeta
        # print self.beta
        # print '==='


class Values_LinearBayesian(Values):
    def __init__(self, l2, s2):
        super(Values_LinearBayesian, self).__init__()
        self.l2 = l2
        self.s2 = s2

        self.A = None
        self.b = None
        self.m = None
        self.S = None

    def value(self, sa):
        if sa.s is tstate:
            return 0.
        if self.m is None:
            return 0.
        return np.dot(sa.phi, self.m)

    def confidence(self, sa):
        if sa.s is tstate:
            return 0.
        if self.S is None:
            return np.sqrt(self.l2 * np.dot(sa.phi, sa.phi))
            # return np.inf
        return np.sqrt(la.multi_dot([sa.phi, self.S, sa.phi]))

    def update_target(self, target, sa):
        if self.A is None or self.b is None:
            ndim = len(sa.phi)
            self.A = np.eye(ndim) / self.l2
            self.b = np.zeros(ndim)

        self.A += np.outer(sa.phi, sa.phi) / self.s2
        self.b += target * sa.phi / self.s2
        self.S = la.inv(self.A)
        self.m = np.dot(self.S, self.b)


# class Values_Approx

# class StateValues(Values):
#     def __init__(self, initv=0., model=None):
#         super(StateValues, self).__init__(initv)
#         self.model = model

#     def optim_action(self, actions, s):
#         if s is tstate:
#             raise ValuesException('No action is available from the tstate state.')
#         if self.model is None:
#             raise ValuesException('This method requires self.model to be set (currently None).')
#         return max(actions, key=lambda a: sum(p * self[s1] for p, s1, _ in self.model.PR_iter(s, a)))


class ActionValues(object):
    # def update(self, method, **kwargs):
    def update(self, r=None, gamma=None, s0=None, a0=None, s1=None, a1=None, actions=None):
        if self.update_method == 'sarsa':
            check = dict(r=r, gamma=gamma, s0=s0, a0=a0, s1=s1, a1=a1)
            update = self.update_sarsa
            args = r, gamma, SAPair(s0, a0), SAPair(s1, a1)
        elif self.update_method == 'qlearning':
            check = dict(r=r, gamma=gamma, s0=s0, a0=a0, s1=s1, actions=actions)
            update = self.update_qlearning
            args = r, gamma, SAPair(s0, a0), s1, actions
        else:
            raise ValuesException('Update target type {} not defined'.format(method))

        args_none = [k for k, v in check.iteritems() if v is None]
        if args_none:
            raise ValuesException('Update target type {} requires {} not be None.'.format(method, args_none))

        update(*args)

    def update_sarsa(self, r, gamma, sa0, sa1):
        target = r + gamma * self.value(sa1)
        self.update_target(target, sa0)

    def update_qlearning(self, r, gamma, sa, s, actions):
        target = r + gamma * self.optim_value(actions, s)
        self.update_target(target, sa)

    def optim_value(self, actions, s):
        # if s is tstate:
        #     return 0.
            # raise ValuesException('No action is available from the tstate state.')
        return max(self.value(SAPair(s, a)) for a in actions)

    def optim_actions(self, actions, s):
        # if s is tstate:
        #     raise ValuesException('No action is available from the tstate state.')
        max_v = self.optim_value(actions, s)
        max_actions = [a for a in actions if self.value(SAPair(s, a)) == max_v]
        return max_actions

    def optim_action(self, actions, s):
        # if s is tstate:
        #     raise ValuesException('No action is available from the tstate state.')
        optim_actions = self.optim_actions(actions, s)
        ai = rnd.choice(len(optim_actions))
        return optim_actions[ai]


class ActionValues_Tabular(ActionValues, Values_Tabular):
    pass

class ActionValues_Linear(ActionValues, Values_Linear):
    pass

class ActionValues_LinearBayesian(ActionValues, Values_LinearBayesian):
    pass



# class ActionValues_Approx(ActionValues):
#     # TODO separation of value from other stats (e.g. number of updates, and learning rate)
#     def __init__(self, env):
#         super(ActionValues_Approx, self).__init__(env)
#         self.params = None
#         self.lr = LearningRate_const(.3)

#     def phi(self, s, a):
#         sphi = s.phi
#         ai = self.env.actions_all.index(a)

#         nactions = len(self.env.actions_all)
#         nsfeats = len(sphi)

#         phi_sa = np.zeros((nactions, nsfeats))
#         phi_sa[ai, :] = sphi
#         return phi_sa.ravel()

#     def value(self, s, a):
#         phi_sa = self.phi(s, a)
#         if self.params is None:
#             self.params = np.zeros(phi_sa.size)

#         return np.dot(self.params, phi_sa.flatten())

#     def update(self, target, s, a):
#         phi_sa = self.phi(s, a)
#         if self.params is None:
#             self.params = np.zeros(phi_sa.size)
#         # np.seterr(all='raise')
#         self.params += self.lr.a * (target - self.value(s, a)) * phi_sa


# class ActionValues_GP(ActionValues):

#     def __init__(self, env):
#         super(ActionValues_GP, self).__init__(env)
#         self.X = None
#         self.Y = None
#         self.gp = None

#     def add_observation(self, x, y):
#         if self.X is None or self.Y is None:
#             self.X = np.atleast_2d(x)
#             self.Y = np.atleast_2d(y)
#         else:
#             self.X = np.vstack((self.X, x))
#             self.Y = np.vstack((self.Y, y))

#         if self.gp is None:
#             ndim = self.X.shape[1]
#             kern = Linear(ndim)
#             self.gp = GPRegression(self.X, self.Y, kernel=kern)

#             # self.gp = GPRegression(self.X, self.Y)
#         else:
#             self.gp.set_XY(self.X, self.Y)

#         try:
#             self.__noptim += 1
#         except AttributeError:
#             self.__noptim = 1
#         if not self.__noptim % 1:
#             self.gp.optimize()

#     def phi(self, s, a):
#         sphi = s.phi
#         ai = self.env.actions_all.index(a)

#         nactions = len(self.env.actions_all)
#         nsfeats = len(sphi)

#         phi_sa = np.zeros((nactions, nsfeats))
#         phi_sa[ai, :] = sphi
#         return phi_sa.ravel()

#     def value(self, s, a):
#         if s is tstate:
#             return 0.
#         if self.gp is None:
#             return 0.
#         phi_sa = self.phi(s, a)
#         phi_sa = np.atleast_2d(phi_sa)
#         m, _ = self.gp.predict(phi_sa)
#         return np.asscalar(m)

#     def update(self, target, s, a):
#         # TODO this has to be wrong... early data was made of bad approximations
#         phi_sa = self.phi(s, a)
#         phi_sa = np.atleast_2d(phi_sa)

#         self.add_observation(phi_sa, target)


# class ActionValues_LinearBayesian(ActionValues):

#     def __init__(self, env, l, s2):
#         super(ActionValues_LinearBayesian, self).__init__(env)
#         self.l = l
#         self.s2 = s2

#         self.A = None
#         self.b = None
#         self.m = None
#         self.S = None

#     def phi(self, s, a):
#         sphi = s.phi
#         ai = self.env.actions_all.index(a)

#         nactions = len(self.env.actions_all)
#         nsfeats = len(sphi)

#         phi_sa = np.zeros((nactions, nsfeats))
#         phi_sa[ai, :] = sphi
#         return phi_sa.ravel()

#     def value(self, s, a):
#         if s is tstate:
#             return 0.
#         if self.m is None:
#             return 0.
#         phi_sa = self.phi(s, a)

#         return np.dot(phi_sa, self.m)

#     def confidence(self, s, a):
#         if s is tstate:
#             return 0.
#         if self.S is None:
#             return np.inf
#         phi_sa = self.phi(s, a)

#         return np.sqrt(la.multi_dot([phi_sa, self.S, phi_sa]))

#     def update(self, target, s, a):
#         phi_sa = self.phi(s, a)

#         if self.A is None or self.b is None:
#             ndim = len(phi_sa)
#             self.A = np.eye(ndim) / self.l
#             self.b = np.zeros(ndim)

#         self.A += np.outer(phi_sa, phi_sa) / self.s2
#         self.b += target * phi_sa / self.s2
#         self.S = la.inv(self.A)
#         self.m = np.dot(self.S, self.b)


# class ActionValues_Linear(ActionValues):

#     def __init__(self, env, l, a):
#         super(ActionValues_Linear, self).__init__(env)
#         self.l = l
#         self.a = a

#         self.beta = None
#         self.elig = None

#     def phi(self, s, a):
#         sphi = s.phi
#         ai = self.env.actions_all.index(a)

#         nactions = len(self.env.actions_all)
#         nsfeats = len(sphi)

#         phi_sa = np.zeros((nactions, nsfeats))
#         phi_sa[ai, :] = sphi
#         return phi_sa.ravel()

#     def value(self, s, a):
#         if s is tstate:
#             return 0.
#         if self.beta is None:
#             return 0.
#         phi_sa = self.phi(s, a)

#         return np.dot(phi_sa, self.beta)

#     def reset(self):
#         self.elig = None

#     def update(self, target, sa):
#         phi_sa = self.phi(sa)
#         ndim = len(phi_sa)

#         if self.beta is None:
#             self.beta = np.zeros(ndim)

#         if self.elig is None:
#             self.elig = np.zeros(ndim)

#         error = target - self.value(s, a)
#         self.elig = self.l * self.elig + phi_sa
#         self.beta += self.a * error * self.elig


# class Values_Linear(Values):

#     def __init__(self, l, a):
#         super(Values_Linear, self).__init__()
#         self.l = l
#         self.a = a

#         self.beta = None
#         # self.elig = None

#     # def phi(self, s, a):
#     #     sphi = s.phi
#     #     ai = self.env.actions_all.index(a)

#     #     nactions = len(self.env.actions_all)
#     #     nsfeats = len(sphi)

#     #     phi_sa = np.zeros((nactions, nsfeats))
#     #     phi_sa[ai, :] = sphi
#     #     return phi_sa.ravel()

#     # TODO some way to compute features...

#     def value(self, s, a):
#         if s is tstate:
#             return 0.
#         if self.beta is None:
#             return 0.
#         phi_sa = self.phi(s, a)

#         return np.dot(phi_sa, self.beta)

# #     def reset(self):
# #         self.elig = None

#     def update(self, target, s, a):
#         phi_sa = self.phi(s, a)
#         ndim = len(phi_sa)

#         if self.beta is None:
#             self.beta = np.zeros(ndim)

#         # if self.elig is None:
#         #     self.elig = np.zeros(ndim)

#         error = target - self.value(s, a)
#         # self.elig = self.l * self.elig + phi_sa
#         # self.beta += self.a * error * self.elig
#         self.beta += self.a * error


# class ActionValues_Linear(ActionValues_Approx, Values_Linear):
#     pass


# class ActionValues_LinearBayesian(ActionValues):

#     def __init__(self, env, l, s2):
#         super(ActionValues_LinearBayesian, self).__init__(env)
#         self.l = l
#         self.s2 = s2

#         self.A = None
#         self.b = None
#         self.m = None
#         self.S = None

#     def phi(self, s, a):
#         sphi = s.phi
#         ai = self.env.actions_all.index(a)

#         nactions = len(self.env.actions_all)
#         nsfeats = len(sphi)

#         phi_sa = np.zeros((nactions, nsfeats))
#         phi_sa[ai, :] = sphi
#         return phi_sa.ravel()

#     def value(self, s, a):
#         if s is tstate:
#             return 0.
#         if self.m is None:
#             return 0.
#         phi_sa = self.phi(s, a)

#         return np.dot(phi_sa, self.m)

#     def confidence(self, s, a):
#         if s is tstate:
#             return 0.
#         if self.S is None:
#             return np.inf
#         phi_sa = self.phi(s, a)

#         return np.sqrt(la.multi_dot([phi_sa, self.S, phi_sa]))

#     def update(self, target, s, a):
#         phi_sa = self.phi(s, a)

#         if self.A is None or self.b is None:
#             ndim = len(phi_sa)
#             self.A = np.eye(ndim) / self.l
#             self.b = np.zeros(ndim)

#         self.A += np.outer(phi_sa, phi_sa) / self.s2
#         self.b += target * phi_sa / self.s2
#         self.S = la.inv(self.A)
#         self.m = np.dot(self.S, self.b)

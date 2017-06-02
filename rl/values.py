from __future__ import division

from contextlib import contextmanager
import math

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from GPy.models import GPRegression
from GPy.kern import Linear
from rl.learning import LearningRate_geom, LearningRate_const

from rl.problems import SAPair

from pytk.util import argmax
from pytk.more_collections import defaultdict_noinsert


class ValuesException(Exception):
    pass


Qtype = object()
Vtype = object()
Atype = object()


class Eligibility(object):
    def __init__(self, vtype, gamma, lambda_):
        self.vtype = vtype
        self.gl = gamma * lambda_
        self.reset()

    def reset(self):
        raise NotImplementedError

    def trace_sa(self, s, a):
        raise NotImplementedError

    def trace_s(self, s):
        return self.trace_sa(s, None)

    def trace(self, *args, **kwargs):
        if self.vtype is Qtype:
            return self.trace_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            return self.trace_s(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) ineligible.'.format(self.vtype))

    def update_sa(self, s, a):
        raise NotImplementedError

    def update_s(self, s):
        return self.update_sa(s, None)

    def update(self, *args, **kwargs):
        if self.vtype is Qtype:
            return self.update_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            return self.update_s(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) ineligible.'.format(self.vtype))

    def __call__(self, *args, **kwargs):
        return self.trace(*args, **kwargs)


class Values(object):

    def __init__(self, vtype):
        self.vtype = vtype

    @contextmanager
    def eligibility(self, gamma, lambda_):
        if hasattr(self, 'elig'):
            raise ValuesException('Eligibility already exists')
        self.elig = self.make_elig(gamma, lambda_)
        yield self.elig
        del self.elig

    def make_elig(self, gamma, lambda_):
        raise NotImplementedError

    @classmethod
    def Q(cls, *args, **kwargs):
        return cls(Qtype, *args, **kwargs)

    @classmethod
    def V(cls, *args, **kwargs):
        return cls(Vtype, *args, **kwargs)

    @classmethod
    def A(cls, *args, **kwargs):
        return cls(Atype, *args, **kwargs)

    def value_sa(self, s, a):
        raise NotImplementedError

    def value_s(self, s):
        return self.value_sa(s, None)

    def value_a(self, a):
        return self.value_sa(None, a)

    def value(self, *args, **kwargs):
        if self.vtype is Qtype:
            return self.value_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            return self.value_s(*args, **kwargs)
        elif self.vtype is Atype:
            return self.value_a(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) unknown.'.format(self.vtype))

    def confidence_sa(self, s, a):
        raise NotImplementedError

    def confidence_s(self, s):
        return self.confidence_sa(s, None)

    def confidence_a(self, a):
        return self.confidence_sa(None, a)

    def confidence(self, *args, **kwargs):
        if self.vtype is Qtype:
            return self.confidence_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            return self.confidence_s(*args, **kwargs)
        elif self.vtype is Atype:
            return self.confidence_a(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) unknown.'.format(self.vtype))

    def update_value_sa(self, s, a, value):
        raise NotImplementedError

    def update_value_s(self, s, value):
        self.update_value_sa(s, None, value)

    def update_value_a(self, a, value):
        self.update_value_sa(None, a, value)

    def update_value(self, *args, **kwargs):
        """ update the value exactly """
        if self.vtype is Qtype:
            self.update_value_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            self.update_value_s(*args, **kwargs)
        elif self.vtype is Atype:
            self.update_value_a(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) unknown.'.format(self.vtype))

    def update_target_sa(self, s, a, target):
        raise NotImplementedError

    def update_target_s(self, s, target):
        self.update_target_sa(s, None, target)

    def update_target_a(self, a, target):
        self.update_target_sa(None, a, target)

    def update_target(self, *args, **kwargs):
        """ update the value such that it tends towards the target value """
        if self.vtype is Qtype:
            self.update_target_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            self.update_target_s(*args, **kwargs)
        elif self.vtype is Atype:
            self.update_target_a(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) unknown.'.format(self.vtype))

    def optim_value_sa(self, s, actions):
        return max(self.value_sa(s, a) for a in actions)

    def optim_value_s(self, s, actions, model):
        def value(s0, a):
            dist_s1 = model.dynamics.dist_s1(s0, a)
            return sum(pr_s1 * (model.task.E_r(s0, a, s1) + model.task.gamma * self.value(s1))
                    for s1, pr_s1 in dist_s1.iteritems())
        return max(value(s, a) for a in actions)

    def optim_value_a(self, actions):
        return self.value_sa(None, actions)

    def optim_value(self, *args, **kwargs):
        if self.vtype is Qtype:
            return self.optim_value_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            return self.optim_value_s(*args, **kwargs)
        elif self.vtype is Atype:
            return self.optim_value_a(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) unknown or unfeasible for this method.'.format(self.vtype))

    def optim_actions_sa(self, s, actions):
        return argmax(lambda a: self.value_sa(s, a), actions, all_=True)

    def optim_actions_s(self, s, actions, model):
        def value(s0, a):
            dist_s1 = model.dynamics.dist_s1(s0, a)
            return sum(pr_s1 * (model.task.E_r(s0, a, s1) + model.task.gamma * self.value(s1))
                    for s1, pr_s1 in dist_s1.iteritems())
        return argmax(lambda a: value(s, a), actions, all_=True)

    def optim_actions_a(self, actions):
        return self.optim_actions_sa(None, actions)

    def optim_actions(self, *args, **kwargs):
        if self.vtype is Qtype:
            return self.optim_actions_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            return self.optim_actions_s(*args, **kwargs)
        elif self.vtype is Atype:
            return self.optim_actions_a(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) unknown or unfeasible for this method.'.format(self.vtype))

    def optim_action_sa(self, s, actions):
        optim_actions = self.optim_actions_sa(s, actions)
        ai = rnd.choice(len(optim_actions))
        return optim_actions[ai]

    def optim_action_s(self, s, actions, model):
        optim_actions = self.optim_actions_s(s, actions, model)
        ai = rnd.choice(len(optim_actions))
        return optim_actions[ai]

    def optim_action_a(self, actions):
        return self.optim_action_sa(None, actions)

    def optim_action(self, *args, **kwargs):
        if self.vtype is Qtype:
            return self.optim_action_sa(*args, **kwargs)
        elif self.vtype is Vtype:
            return self.optim_action_s(*args, **kwargs)
        elif self.vtype is Atype:
            return self.optim_action_a(*args, **kwargs)
        else:
            raise ValuesException('ValueType ({}) unknown or unfeasible for this method.'.format(self.vtype))

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class Values_Tabular(Values):
    def __init__(self, vtype):
        super(Values_Tabular, self).__init__(vtype)
        self.values = defaultdict_noinsert(float)  # float() is equivalent to lambda: 0.

    @property
    def initvalue(self):
        return self.values.default_factory()

    @initvalue.setter
    def initvalue(self, value):
        self.values.default_factory = lambda: value

    def value_sa(self, s, a):
        return self.values[s, a]

    def update_value_sa(self, s, a, value):
        self.values[s, a] = float(value)


class Values_TabularCounted(Values_Tabular):
    class Eligibility_TabularCounted(Eligibility):
        def reset(self):
            self.traces = defaultdict_noinsert(float)  # float() is equivalent to lambda: 0

        def trace_sa(self, s, a):
            return self.traces[s, a]

        def update_sa(self, s, a):
            for sa in self.traces.viewkeys():
                self.traces[sa] *= self.gl
            self.traces[s, a] += 1


    def __init__(self, vtype, alpha=None):
        super(Values_TabularCounted, self).__init__(vtype)

        if alpha is None:
            def alpha(k):  return 1 / (k + 1)

        self.alpha = alpha
        self.counter = defaultdict_noinsert(int)  # int() is equivalent to lambda: 0

    def make_elig(self, gamma, lambda_):
        return type(self).Eligibility_TabularCounted(self.vtype, gamma, lambda_)

    def value_sa(self, s, a):
        value = super(Values_TabularCounted, self).value_sa(s, a)
        return value

    def confidence_sa(self, s, a):
        n_sa = self.counter[s, a]
        n_s = self.counter[s, None]

        try:
            _2logn = 2 * math.log(n_s)
        except ValueError:
            _2logn = -np.inf
        try:
            _2logn_div_n = _2logn / n_sa
        except ZeroDivisionError:
            return np.inf
        return math.sqrt(_2logn_div_n)

    def update_value_sa(self, s, a, value):
        super(Values_TabularCounted, self).update_value_sa(s, a, value)
        if s is not None and a is not None:
            self.counter[s, a] += 1
        if s is not None:
            self.counter[s, None] += 1
        if a is not None:
            self.counter[None, a] += 1
        self.counter[None, None] += 1

    def update_target_sa(self, s, a, target):
        v = self.value_sa(s, a)
        n = self.nupdates_sa(s, a)

        try:
            e = self.elig.trace_sa(s, a)
        except AttributeError:
            dv = self.alpha(n) * (target - v)
        else:
            dv = self.alpha(n) * (target - v) * e

        value = v + dv
        self.update_value_sa(s, a, value)

    def nupdates_sa(self, s, a):
        return self.counter[s, a]

    def nupdates_s(self, s):
        return self.counter[s, None]

    def nupdates_a(self, a):
        return self.counter[None, a]

    def nupdates(self):
        return self.counter[None, None]


class Values_Linear(Values):
    class Eligibility_Linear(Eligibility):
        def reset(self):
            self.traces = None

        def trace_sa(self, s, a):
            if self.traces is None:
                self.traces = np.zeros(len(sa.phi))
            return self.traces

        def update_sa(self, s, a):
            sa = SAPair(s, a)
            if self.traces is None:
                self.traces = np.zeros(len(sa.phi))
            else:
                self.traces *= self.gl
            self.traces += sa.phi


    def __init__(self, vtype, alpha=1.):
        super(Values_Linear, self).__init__(vtype)
        self.alpha = alpha
        self.beta = None

    def make_elig(self, gamma, lambda_):
        return type(self).Eligibility_Linear(self.vtype, gamma, lambda_)

    def value_sa(self, s, a):
        if self.beta is not None:
            # TODO eradicate SAPair
            sa = SAPair(s, a)
            return np.dot(sa.phi, self.beta)

    def update_target_sa(self, s, a, target):
        sa = SAPair(s, a)
        if self.beta is None:
            self.beta = np.zeros(len(sa.phi))
        self.beta += self.alpha * (target - self.value_sa(s, a)) * sa.phi


class Values_LinearBayesian(Values):
    class Eligibility_LinearBayesian(Eligibility):
        def reset(self):
            self.traces = None

        def trace_sa(self, s, a):
            if self.traces is None:
                self.traces = np.zeros(len(sa.phi))
            return self.traces

        def update_sa(self, s, a):
            sa = SAPair(s, a)
            if self.traces is None:
                self.traces = np.zeros(len(sa.phi))
            else:
                self.traces *= self.gl
            self.traces += sa.phi


    def __init__(self, vtype, l2, s2):
        super(Values_LinearBayesian, self).__init__(vtype)
        self.l2 = l2
        self.s2 = s2

        self.A = None
        self.b = None
        self.m = None
        self.S = None

    def make_elig(self, gamma, lambda_):
        return type(self).Eligibility_LinearBayesian(self.vtype, gamma, lambda_)

    def value_sa(self, s, a):
        if self.m is not None:
            # TODO eradicate SAPair
            sa = SAPair(s, a)
            return np.dot(sa.phi, self.m)

    def confidence_sa(self, s, a):
        # TODO eradicate SAPair
        sa = SAPair(s, a)
        if self.S is not None:
            return np.sqrt(la.multi_dot([sa.phi, self.S, sa.phi]))
        return np.sqrt(self.l2 * np.dot(sa.phi, sa.phi))

    def update_target_sa(self, s, a, target):
        # TODO eradicate SAPair
        sa = SAPair(s, a)

        if self.A is None or self.b is None:
            ndim = len(sa.phi)
            self.A = np.eye(ndim) / self.l2
            self.b = np.zeros(ndim)

        self.A += np.outer(sa.phi, sa.phi) / self.s2
        self.b += target * sa.phi / self.s2
        self.S = la.inv(self.A)
        self.m = np.dot(self.S, self.b)


class Values_GP(Values):
    def __init__(self, vtype, optim=True):
        super(Values_LinearBayesian, self).__init__(vtype)
        self.optim = optim

        self.X = None
        self.Y = None
        self.gp = None

    def add_observation(self, x, y):
        try:
            self.X = np.vstack((self.X, x))
            self.Y = np.vstack((self.Y, y))
        except ValueError:  # None in (self.X, self.Y)
            self.X = np.atleast_2d(x)
            self.Y = np.atleast_2d(y)

        try:
            self.gp.set_XY(self.X, self.Y)
        except AttributeError:  # self.gp is None
            ndim = self.X.shape[1]
            kern = Linear(ndim)
            self.gp = GPRegression(self.X, self.Y, kernel=kern)
        finally:
            if self.optim:
                self.gp.optimize()

    def value_sa(self, s, a):
        if self.gp is not None:
            # TODO eradicate SAPair
            sa = SAPair(s, a)
            sa_phi = np.atleast_2d(sa.phi)
            m, _ = self.gp.predict(sa_phi)
            return np.asscalar(m)

    def confidence_sa(self, s, a):
        # TODO implement
        raise NotImplementedError

    def update_target_sa(self, s, a, target):
        # TODO eradicate SAPair
        sa = SAPair(s, a)
        self.add_observation(sa.phi, target)


class Values_Preference(Values_Tabular):
    """ Special case tabular values for softmax policy """
    def __init__(self, vtype, alpha, beta, ref=0.):
        super(Values_Preference, self).__init__(vtype)
        self.alpha = alpha
        self.beta = beta
        self.ref = ref

    def update_target_sa(self, s, a, r):
        rdiff = r - self.ref
        value = self.value_sa(s, a) + self.beta * rdiff
        super(Values_Preference, self).update_value_sa(s, a, value)
        self.ref += self.alpha * rdiff


class Values_Value2Value(Values):
    def __init__(self, vtype, ref, sys):
        super(Values_Value2Value, self).__init__(vtype)
        self.ref = ref
        self.sys = sys


class Values_V2Q(Values_Value2Value):
    def __init__(self, vtype, ref, sys, model):
        super(Values_V2Q, self).__init__(vtype, ref, sys)
        self.model = model

    def value_sa(self, s0, a):
        dist_s1 = self.model.dynamics.dist_s1(s0, a)
        return sum(pr_s1 * (self.model.task.E_r(s0, a, s1) + self.model.task.gamma * self.ref(s1)) \
                for s1, pr_s1 in model_dist.iteritems())


class Values_Q2V(Values_Value2Value):
    def __init__(self, vtype, ref, sys, policy):
        super(Vaues_Q2V, self).__init__(vtype, ref, sys)
        self.policy = policy

    def value_sa(self, s, _):
        actions = self.sys.actions(s)
        policy_dist = self.policy.dist_sa(s, actions)
        return sum(pr_a * self.ref(s, a) for a, pr_a in policy_dist.iteritems())

import numpy as np

taction = type(
    'TerminalAction',
    (object,),
    dict(__str__=lambda self: 'A(Terminal)'),
)()


class State(object):
    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self == other


class Action(object):
    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self == other


class SAPairException(Exception):
    pass


class SAPair(object):
    def __init__(self, s=None, a=None):
        if s is None and a is None:
            raise SAPairException('Either s or a has to be given')
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

    def __hash__(self):
        return hash((self.s, self.a))

    def __eq__(self, other):
        return self.s == other.s and self.a == other.a

    def __ne__(self, other):
        return not self == other

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



































class Model(object):
    """ State and Reward dynamics class

    Represents real and learned problem dynamics.
    """

    def pr_s0(self, s0=None):
        raise NotImplementedError

    def pr_s1(self, s0, a, s1=None):
        raise NotImplementedError

    def E_r(self, s0, a, s1):
        pass

    # def pr_rs1(self, s0, a, s1=None):
    #     if s1 is not None:

    # def pr_r(self, s0, a, s1, r=None):
    #     raise NotImplementedError

    # def pr_rs1(self, s0, a, rs1=None):
    #     raise NotImplementedError

    def sample_s0(self):
        raise NotImplementedError

    def sample_s1(self, s0, a):
        raise NotImplementedError

    def sample_r(self, s0, a, s1):
        raise NotImplementedError

    def sample_rs1(self, s0, a):
        s1 = self.sample_s1(s0, a)
        r = self.sample_r(s0, a, s1)
        return r, s1


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

import numpy.random as rnd


class ModelException(Exception):
    pass


class model(object):
    """ Contains environment dynamics """
    def __init__(self, env):
        self.env = env
    
    def P(self, a, s0, s1):
        raise NotImplementedError

    def R(self, a, s0, s1):
        raise NotImplementedError

    def PR(self, a, s0, s1):
        return self.P(a, s0, s1), self.R(a, s0, s1)

    def PR_iter(self, s0, a):
        for s1 in self.env.states():
            p, r = self.PR(a, s0, s1)
            if p:
                yield p, s1, r

    def sample_s1(self, a, s0):
        ps, s1s = [], []
        for p, s1, _ in self.PR_iter(a, s0):
            ps.append(p)
            s1s.append(s1)

        i = rnd.choice(len(ps), p=ps)
        return s1s[i]
    
    def sample_r(self, a, s0):
        raise NotImplementedError

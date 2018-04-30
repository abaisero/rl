from .value import Q


# TODO readonly
class VBased(Q):

    def __init__(self, V, model, gamma):
        super().__init__(V.env)
        self.V = V
        self.model = model
        self.gamma = gamma

    def value(self, s, a):
        dist = self.model.s1.dist(s, a)
        return sum(self.model.r.E(s, a, s1) + self.gamma * prs1 * self.V(s1) for s1, prs1 in dist)

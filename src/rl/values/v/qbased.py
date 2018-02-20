from .value import V


class QBased(V):
    # TODO readonly

    def __init__(self, Q, policy):
        super().__init__(Q.env)
        self.Q = Q
        self.policy = policy

    # def value(self, s):
    #     # return np.dot(self.Q[s, :], self.policy.dist[s, :])
    #     return sum(pra * self.Q[s, a] for a, pra in self.policy.dist(s).items())
    #     # return sum(pra * self.Q(s, a) for a, pra in self.policy.dist(s).items())

    def __getitem__(self, s):
        dist = self.policy.dist(s)
        return sum(pra * self.Q[s, a] for a, pra in dist)

    # # TODO readonly
    # def update_value(self, s, value):
    #     raise NotImplementedError

    # def update_target(self, s, target):
    #     raise NotImplementedError

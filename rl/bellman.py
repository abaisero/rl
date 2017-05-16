class Bellman(object):
    def __init__(self, sys, model, policy=None):
        self.sys = sys
        self.model = model
        self.policy = policy

    @staticmethod
    def Q(Q, sys, model):
        return Bellman_Q(Q, sys, model)

    @staticmethod
    def V(V, sys, model):
        return Bellman_V(V, sys, model)

    def __call__(self, *args, **kwargs):
        return self.rhs(*args, **kwargs)


class Bellman_Q(Bellman):
    def __init__(self, Q, sys, model, policy=None):
        super(Bellman_Q, self).__init__(sys, model, policy)
        self.Q = Q

    def __rhs(self, s0, value):
        model_dist = self.model.dynamics.pr_s1(s0, a)
        return sum(pr_s1 * (self.model.task.E_r(s0, a, s1) + self.model.task.gamma * value(s1))
                for s1, pr_s1 in model_dist.viewitems())

    def __rhs_policy(self, s0):
        def value(s1):
            actions1 = self.sys.actions(s1)
            return self.Q.optim_value(s1, actions1)
        return self.__rhs(s0, value)

    def __rhs_optim(self, s0):
        def value(s1):
            actions1 = self.sys.actions(s1)
            policy_dist = self.policy.dist(s1, actions1)
            return sum(pr_a1 * self.Q(s1, a1) for a1, pr_a1 in policy_dist.viewitems())
        return self.__rhs(s0, value)

    def rhs(self, s, a, optim=False):
        return (self.__rhs_optim(s, a)
                if optim
                else self.__rhs_policy(s, a))


class Bellman_V(Bellman):
    def __init__(self, V, sys, model, policy=None):
        super(Bellman_V, self).__init__(sys, model, policy)
        self.V = V

    def __value(self, s0, a):
        dist_s1 = self.model.dynamics.dist_s1(s0, a)
        return sum(pr_s1 * (self.model.task.E_r(s0, a, s1) + self.model.task.gamma * self.V(s1))
                for s1, pr_s1 in dist_s1.viewitems())

    def __rhs_policy(self, s):
        actions = self.sys.actions(s)
        policy_dist = self.policy.dist(s, actions)
        return sum(pr_a * self.__value(s, a) for a, pr_a in policy_dist.viewitems())

    def __rhs_optim(self, s):
        return max(self.__value(s, a) for a in self.sys.actions(s))

    def rhs(self, s, optim=False):
        return (self.__rhs_optim(s)
                if optim
                else self.__rhs_policy(s))

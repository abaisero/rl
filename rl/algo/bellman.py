from rl.problems import SAPair


class bellman(object):
    def __init__(self, sys, model, gamma):
        self.sys = sys
        self.model = model
        self.gamma = gamma

    def update(self, *args, **kwags):
        raise NotImplementedError

    def optim(self, *args, **kwargs):
        raise NotImplementedError


class bellman_statevalues(bellman):
    def __init__(self, env, model, gamma, V):
        super(bellman_statevalues, self).__init__(env, model, gamma)
        self.V = V

    def __update(self, s0, a):
        return sum(self.model.P(a, s0, s1) * (self.model.R(a, s0, s1) + self.gamma * self.V[s1]) for s1 in self.env.states(terminal=True))

    def update(self, s, policy):
        raise NotImplementedError
        return sum(p * self.__update(s, a) for a, p in policy[s])

    def optim(self, s):
        return max(self.__update(s, a) for a in self.env.actions(s))

    def optim_argmax(self, s):
        return max(self.env.actions(s), key=lambda a: self.__update(s, a))


class bellman_statevalues(bellman):
    def __init__(self, sys, model, gamma, V):
        super(bellman_statevalues, self).__init__(sys, model, gamma)
        self.V = V

    def rhs(self, s0, a):
        pr_s1 = np.array([self.model.pr_s1(s0, a, s1) for s1 in self.sys.statelist])
        E_r = np.array([self.model.E_r(s0, a, s1) for s1 in self.sys.statelist])
        V_s1 = np.array([self.V(s1) for s1 in self.sys.statelist])
        return np.dot(pr_s1, E_r + self.gamma * V_s1)
        # return sum(self.model.P(a, s0, s1) * (self.model.R(a, s0, s1) + self.gamma * self.V[s1]) for s1 in self.sys.statelist)

    def update(self, s, policy):
        return sum(p * self.rhs(s, a) for a, p in policy.pr_a(s).iteritems())

    def optim(self, s):
        return max(self.rhs(s, a) for a in self.sys.actions(s))

    def optim_argmax(self, s):
        return max(self.sys.actions(s), key=lambda a: self.rhs(s, a))


# TODO this is closely related with the values functions
def rhs(sys, s0, a, V, gamma):
    return sum(pr_s1 * (sys.model.E_r(s0, a, s1) + gamma * V(SAPair(s1)))
            for s1, pr_s1 in sys.model.pr_s1(s0, a).iteritems())
    # expected_Vs1 = 0
    # for s1, p in sys.model.pr_s1(s0, a).iteritems():
    #     expected_Vs1 += p * (sys.model.E_r(s0, a, s1) + gamma * V(s1))
    # return sum(expected_Vs1)

    # pr_s1 = np.array([model.pr_s1(s0, a, s1) for s1 in states])
    # E_r = np.array([model.E_r(s0, a, s1) for s1 in states])
    # V_s1 = np.array([V(s1) for s1 in states])
    # return np.dot(pr_s1, E_r + gamma * V_s1)


def equation(s, model, policy):
    pass
    # return sum(p * rhs(s, a) for a, p in policy.pr(s).itervalues())


def equation_optim(sys, s, V, gamma):
    return max(rhs(sys, s, a, V, gamma) for a in sys.actions(s))






# class bellman_actionvalues(bellman):
#     def __init__(self, env, model, gamma, Q):
#         super(bellman_statevalues, self).__init__(env, model, gamma)
#         self.Q = Q

#     def update_(self, s0, a, V):
#         return sum(self.model.P(a, s0, s1) * (self.model.R(a, s0, s1) + self.gamma * V[s1]) for s1 in self.env.states())

#     def update(self, s0, a, policy):
#         def V(s):
#             return sum(p * self.Q[s, a] for p, a in policy[s0])
#         return self.update_(s0, a, V)

#     def update_optim(self, s0, with_argmax=False):
#         def V(s):
#             return sum(p * self.Q[s, a] for p, a in policy[s0])
#         max_, argmax_ = max(((self.update_(s0, a, V), a) for a in self.env.actions(s0)), key=lambda ba: b[0])
#         return (max_, argmax_) if with_argmax else max_

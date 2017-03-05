class bellman(object):
    def __init__(self, env, model, gamma):
        self.env = env
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

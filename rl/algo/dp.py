from rl.problems import SAPair

import bellman


class policy_iteration(object):
    def __init__(self, env, model, policy, V, gamma):
        self.env = env
        self.model = model
        self.policy = policy
        self.V = V
        self.gamma = gamma

        self.bellman = bellman_statevalues(env, model, gamma, V)

    def evaluation(self):
        delta = 1
        while 1e-8 < delta:
            delta = 0
            for s in self.env.states():
                v = self.bellman.optim(s)
                delta = max(delta, abs(v - self.V[s]))
                self.V[s] = v

    def improvement(self):
        policy_stable = True
        for s in self.env.states():
            a = self.bellman.optim_argmax(s)
            policy_stable = policy_stable and ( a == self.policy[s] )
            self.policy[s] = a
        return policy_stable

    def iteration(self):
        policy_stable = True
        while policy_stable:
            self.evaluation()
            policy_stable = self.improvement()


class value_iteration(object):
    def __init__(self, env, model, policy, V, gamma):
        self.env = env
        self.model = model
        self.policy = policy
        self.V = V
        self.gamma = gamma

        self.bellman = bellman_statevalues(env, model, gamma, V)

    def iteration(self):
        delta = 1
        while 1e-8 < delta:
            delta = 0
            for s in self.env.states():
                v = self.bellman.optim(s)
                delta = max(delta, abs(v - self.V[s]))
                self.V[s] = v

            print [self.V[s] for s in self.env.states()]

        for s in self.env.states():
            self.policy[s] = self.bellman.optim_argmax(s)


def value_iteration(sys, V, gamma):
    delta = 1
    while delta > 1e-8:
        delta = 0
        for s in sys.statelist:
            v = bellman.equation_optim(sys, s, V, gamma)
            delta = max(delta, abs(v - V(SAPair(s))))
            V.update(v, SAPair(s))

import numpy as np


class simulator(object):
    def __init__(self, env, mod, policy):
        self.env = env
        self.mod = mod
        self.policy = policy

    def run(self):
        s0 = self.mod.sample_s0()
        print 'run start: {}'.format(s0)
        episode = []

        while s0 not in self.env.states(begin=False, middle=False, terminal=True):
            # print '---'
            # print 's0: {}'.format(s0)
            a = self.policy.sample_a(s0)
            # print 'a: {}'.format(a)
            s1 = self.mod.sample_s1(a, s0)
            # print 's1: {}'.format(s1)
            r = self.mod.R(a, s0, s1)
            # print 'r: {}'.format(r)

            episode.append((s0, a, r, s1))
            s0 = s1

        return episode


class on_policy_mc(object):
    def __init__(self, env, mod, policy, Q, gamma):
        self.env = env
        self.mod = mod
        self.policy = policy
        self.Q = Q
        self.gamma = gamma

        self.sim = simulator(env, mod, policy)

    def run(self):
        # initialization
        returns = dict()

        # while True:
        for i in range(1000):
            # print 'running episode {}'.format(i)

            # a) generate episode
            episodes = self.sim.run()
            episodes_states = set()

            # b) update Q
            episodes_returns = dict()
            R = 0
            for s0, a, r, s1 in episodes[::-1]:
                R = self.gamma * R + r
                episodes_returns[s0, a] = R

                episodes_states.add(s0)

            for (s0, a), R in episodes_returns.iteritems():
                returns.setdefault((s0, a), []).append(R)
                self.Q[s0, a] = np.mean(returns[s0, a])

            # c) update policy
            for s in episodes_states:
                self.policy[s] = self.Q.optim_action(s)

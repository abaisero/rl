
class POMCP(object):
    def __init__(self, pomdp, model, policy_tree, policy_rollout, Q, gamma=1):
        self.max_depth = log(eps) / log(gamma)

        self.policy_tree = policy_tree
        self.policy_rollout = policy_rollout

    def select_action(self, sroot, budget, verbose=False):
        pass

    def simulate(self, s, h, depth):
        if depth > self.max_depth:
            return 0.
        if h not in tree?

    def rollout(self, s, h, depth):
        if depth > self.max_depth:
            return 0.
        a = self.policy_rollout(h)
        s1, o, r = model.sample_sor(s, a)
        h.add(a, o)
        return r + gamma * self.rollout(s1, h, depth+1)

from rl.problems import SAPair


class Agent(object):
    def sample_a(self):
        pass

class BanditAgent(Agent):
    def __init__(self, mab, Q, policy):
        self.mab = mab
        self.Q = Q
        self.policy = policy

    def sample_b(self):
        return self.policy.sample_a(self.mab.actionlist)

    def sample_br(self):
        b = self.sample_b()
        r = b.sample_r()
        return b, r

    def feedback(self, a, r):
        self.Q.update_target(r, SAPair(None, a))


# TODO sometimes agent and policy are different things.. sometimes they are the
# same
# NOTE what if I give the possible actions to the policy

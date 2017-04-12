from rl.problems import SAPair


class Agent(object):
    def sample_a(self):
        pass

class BanditAgent(Agent):
    def __init__(self, mab, Q, policy):
        self.mab = mab
        self.Q = Q
        self.policy = policy

    def sample_a(self):
        return self.policy.sample_a(self.mab.actionlist)

    def feedback(self, a, r):
        self.Q.update(r, SAPair(None, a))


# TODO sometimes agent and policy are different things.. sometimes they are the
# same
# NOTE what if I give the possible actions to the policy

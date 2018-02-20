from rl.problems import SAPair


# TODO As opposed to a policy, an agent has an identity--typically but not
# necessarily sufficiently a name--and may be able to collect statistics about
# its own existence and experience
class Agent:
    def __init__(self, name, policy, feedback=None):
        self.name = name
        self.policy = policy
        self.feedback = feedback

    def __str__(self):
        return self.name

    # TODO how to reset agent?


# TODO scrap the next

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
        self.Q.update_target(SAPair(a=a), r)

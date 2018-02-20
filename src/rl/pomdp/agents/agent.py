import rl.misc as rlmisc


class Agent(rlmisc.Agent):
    def act(self, context):
        return self.policy.sample()

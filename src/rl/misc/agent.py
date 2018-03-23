class Agent:
    def __init__(self, name, policy):
        self.name = name
        self.policy = policy

    def reset(self):
        self.policy.reset()

    def restart(self):
        self.policy.restart()

    def act(self, context):
        raise NotImplementedError

    def feedback(self, context, a, feedback, context1):
        pass

    def feedback_episode(self, episode):
        pass

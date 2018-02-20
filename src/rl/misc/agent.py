class Agent:
    def __init__(self, name, env, policy):
        self.name = name
        self.env = env
        self.policy = policy

    def reset(self):
        self.policy.reset()

    def restart(self):
        self.policy.restart()

    def act(self, context):
        raise NotImplementedError

    def feedback(self, sys, context, a, feedback):
        pass

    def feedback_episode(self, sys, episode):
        pass

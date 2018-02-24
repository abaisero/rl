class P:
    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        pass

    def restart(self):
        pass

    def feedback(self, sys, context, a, feedback):
        pass

    def feedback_episode(self, sys, episode):
        pass

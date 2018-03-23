class P:
    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        pass

    def restart(self):
        pass

    def feedback(self, context, a, feedback, context1):
        pass

    def feedback_episode(self, episode, context1):
        pass

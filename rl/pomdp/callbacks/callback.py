class Callback:
    def __init__(self, env):
        self.env = env

    def feedback(self, sys, context, a, feedback):
        pass

    def feedback_episode(self, sys, episode):
        pass

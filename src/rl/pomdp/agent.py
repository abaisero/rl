import logging
logger = logging.getLogger(__name__)

# import rl.misc as rlmisc


# class Agent(rlmisc.Agent):
#     logger = logging.getLogger(f'{__name__}.Agent')

#     def act(self, context):
#         a = self.policy.sample([])  # TODO fill in real pcontext
#         self.logger.debug(f'act({context}) -> {a}')
#         return a


class Agent:
    logger = logging.getLogger(f'{__name__}.Agent')

    def __repr__(self):
        if self.name is None:
            return f'Agent(algo={self.algo}, policy={self.policy})'
        return f'Agent({self.name})'

    def __init__(self, policy, algo, *, name=None):
        self.policy = policy
        self.algo = algo
        self.name = name

    def reset(self):
        self.policy.reset()
        self.algo.reset()

    def restart(self):
        self.pcontext = self.policy.new_pcontext()
        self.algo.restart()

    def act(self, econtext):
        return self.policy.sample(self.pcontext)

    def feedback(self, context, a, feedback, context1):
        self.algo.feedback(self.pcontext, context, a, feedback, context1)

    def feedback_episode(self, episode):
        self.algo.feedback_episode(self.pcontext, episode)

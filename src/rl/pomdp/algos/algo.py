import logging
logger = logging.getLogger(__name__)

import rl.misc as rlmisc


# class Agent(rlmisc.Agent):
#     logger = logging.getLogger(f'{__name__}.Agent')

#     def act(self, context):
#         a = self.policy.sample([])  # TODO fill in real pcontext
#         self.logger.debug(f'act({context}) -> {a}')
#         return a


class Algo:
    logger = logging.getLogger(f'{__name__}.Algo')

    # def __init__(self, name, policy, algo=None):
    #     self.name = name
    #     self.policy = policy
    #     self.algo = algo

    # def reset(self):
    #     self.policy.reset()
    #     self.algo.reset()
    #     self.pcontext = self.policy.new_pcontext()

    # def restart(self):
    #     self.policy.restart()
    #     self.algo.restart()

    # def act(self, econtext):
    #     return self.policy.sample([])

    # def feedback(self, context, a, feedback, context1):
    #     self.algo.feedback(self.pcontext, context, a, feedback, context1)

    # def feedback_episode(self, episode):
    #     self.algo.feedback_episode(self.pcontext, episode)

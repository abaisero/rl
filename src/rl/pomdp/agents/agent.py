import logging
logger = logging.getLogger(__name__)

import rl.misc as rlmisc


class Agent(rlmisc.Agent):
    logger = logging.getLogger(f'{__name__}.Agent')

    def act(self, context):
        a = self.policy.sample()
        self.logger.debug(f'act({context}) -> {a}')
        return a

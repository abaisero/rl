import logging
logger = logging.getLogger(__name__)

import rl.misc as rlmisc


class Agent(rlmisc.Agent):
    def act(self, context):
        return self.policy.sample(context.s)


class PreAgent(Agent):
    def restart(self):
        self.__precontext = None
        self.__preaction = None

    def act(self, context, preact=False):
        if self.__precontext is not None and self.__precontext == context:
            a = self.__preaction
            if not preact:
                self.__precontext = None
                self.__preaction = None
            return a

        if self.__precontext is not None and self.__precontext != context:
            logger.warning(f'Stored context {self.__precontext} does not match given one {context};  ignoring stored context.')

        a = super().act(context)
        if preact:
            self.__precontext = context
            self.__preaction = a
        else:
            self.__precontext = None
            self.__preaction = None
        return a

import logging
logger = logging.getLogger(__name__)

import rl.misc as rlmisc

from collections import namedtuple


Context = namedtuple('Context', 't')
Feedback = namedtuple('Feedback', 'r, o')


class System(rlmisc.System):
    logger = logging.getLogger(f'{__name__}.System')

    @property
    def context(self):
        return Context(self.t)

    def step(self, a):
        s = self.s

        s1, = self.model.s1.sample(s, a)
        o, = self.model.o.sample(s, a, s1)
        r = self.model.r.sample(s, a, s1)

        self.logger.debug(f'step() \t; s={s} \t; a={a} \t; s1={s1} \t; o={o} \t; r={r}')

        self.s = s1
        self.terminal = False

        return Feedback(r, o)

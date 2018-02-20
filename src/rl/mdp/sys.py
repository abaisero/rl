import rl.misc as rlmisc

from collections import namedtuple


Context = namedtuple('Context', 't, s')
Feedback = namedtuple('Feedback', 'r, s1')


class System(rlmisc.System):
    @property
    def context(self):
        return Context(self.t, self.s)

    def step(self, a):
        s = self.s
        s1, = self.model.s1.sample(s, a)
        r = self.model.r.sample(s, a, s1)

        self.s = s1
        self.terminal = False

        return Feedback(r, s1)

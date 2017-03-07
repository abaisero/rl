from pytk.decorators import memoizemethod


class environment(object):
    def states(self, begin=True, middle=True, terminal=False):
        raise NotImplementedError

    @memoizemethod
    def nstates(self, begin=True, middle=True, terminal=False):
        return len(self.states(begin=begin, middle=middle, terminal=terminal))

    def actions(self, s):
        return NotImplementedError

    @memoizemethod
    def nactions(self, s):
        return len(self.actions(s))

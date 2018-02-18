from .policy import Policy

import numpy as np
import numpy.random as rnd

from scipy.misc import logsumexp


class Softmax(Policy):
    def prefs(self, s=None, a=None):
        raise NotImplementedError

    def indices(self, *items):
        return tuple(slice(None) if item is None else item.i for item in items)

    def logprobs(self, s=None, a=None):
        idx = self.indices(s, a)

        prefs = self.prefs()
        logprobs = prefs - logsumexp(prefs, axis=1, keepdims=True)
        return logprobs[idx]

    def probs(self, s=None, a=None):
        idx = self.indices(s, a)

        logprobs = self.logprobs()
        probs = np.exp(logprobs)
        return probs[idx]


    def dist(self, s):
        probs = self.probs(s)
        for a in self.env.actions:
            yield a, probs[a.i]

    def pr(self, s, a):
        probs = self.probs(s)
        return probs[a.i]

    def sample(self, s):
        probs = self.probs(s)
        return rnd.choice(list(self.env.actions), p=probs)

from .policy import Policy


class Softmax(Policy):
    def prefs(self, a=None):
        raise NotImplementedError

    def indices(self, *items):
        return tuple(slice(None) if item is None else item.i for item in items)

    def logprobs(self, a=None):
        idx = self.indices(a)

        prefs = self.prefs()
        logprobs = prefs - logsumexp(prefs, axis=1)[: np.

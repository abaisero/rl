from .softmax import Softmax


class QSoftmax(Softmax):
    def __init__(self, env, Q):
        super().__init__(env)
        self.Q = Q

    def prefs(self, s=None, a=None):
        idx = self.indices
        si = slice(None) if s is None else s.i
        ai = slice(None) if a is None else a.i

        return self.Q[si, ai]

#         return np.array([self.Q(s, a) for a in self.env.actions])

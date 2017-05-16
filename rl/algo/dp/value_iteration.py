from rl.bellman import Bellman


class ValueIteration(object):
    def __init__(self, sys, model):
        self.sys = sys
        self.model = model

    @classmethod
    def Q(self, Q, sys, model):
        return ValueIteration_Q(Q, sys, model)

    @classmethod
    def V(self, V, sys, model):
        return ValueIteration_V(V, sys, model)

    def run(self):
        raise NotImplementedError


class ValueIteration_Q(ValueIteration):
    def __init__(self, Q, sys, model):
        super(ValueIteration_Q, self).__init__(sys, model)
        self.Q = Q
        self.bellman = Bellman.Q(Q, sys, model)

    def run(self):
        delta = 1
        while delta > 1e-8:
            delta = 0
            for s in self.sys.states():
                for a in self.sys.actions(s):
                    q = self.bellman(s, a, optim=True)
                    delta = max(delta, abs(q - self.Q(s, a)))
                    self.Q.update_value(s, q)


class ValueIteration_V(ValueIteration):
    def __init__(self, V, sys, model):
        super(ValueIteration_V, self).__init__(sys, model)
        self.V = V
        self.bellman = Bellman.V(V, sys, model)

    def run(self):
        delta = 1
        while delta > 1e-8:
            delta = 0
            for s in self.sys.states():
                v = self.bellman(s, optim=True)
                delta = max(delta, abs(v - self.V(s)))
                self.V.update_value(s, v)

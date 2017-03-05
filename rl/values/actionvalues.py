class actionvalues(object):
    def __init__(self, states, actions, init=0):
        self.states = states
        self.actions = actions
        self.Q = {(s, a): init for s in states for a in actions(s)}

    def __getitem__(self, sa):
        return self.Q[sa]

    def __setitem__(self, sa, q):
        self.Q[sa] = q

    def max_action(self, s, with_argmax=False):
        maxq, maxa = max(((self[s, a], a) for a in self.actions(s)), key=lambda qa: q[0])
        return maxq, maxa if with_argmax else maxq

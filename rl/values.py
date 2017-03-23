from rl.learning import LearningRate_geom


class ValuesException(Exception):
    pass


class Values(object):
    def __init__(self, env, LR=None, initv=0.):
        if LR is None:
            LR = LearningRate_geom()

        self.env = env
        self.LR = LR
        self.initv = initv

        self.vdict = {}

    @property
    def initvnlr(self):
        return self.initv, 0, self.LR()

    def vnlr(self, s, a=None):
        if s is self.env.terminal:
            return 0., 0, None
        return self.vdict.get((s, a), self.initvnlr)

    def v(self, s, a=None):
        return self.vnlr(s, a)[0]

    def n(self, s, a=None):
        return self.vnlr(s, a)[1]

    def lr(self, s, a=None):
        return self.vnlr(s, a)[2]

    def update(self, target, s, a=None):
        v, n, lr = self.vnlr(s, a)
        v += lr.a * (target - self.v(s, a))
        n += 1
        lr.update()
        self.vdict[s, a] = v, n, lr

    def __call__(self, s, a=None):
        return self.v(s, a)


class StateValues(Values):
    def __init__(self, env, initv=0., model=None):
        super(StateValues, self).__init__(env, initv)
        self.model = model

    def optim_action(self, s):
        if s is self.env.terminal:
            raise ValuesException('No action is available from the terminal state.')
        if self.model is None:
            raise ValuesException('This method requires self.model to be set (currently None).')
        return max(self.env.actions(s), key=lambda a: sum(p * self[s1] for p, s1, _ in self.model.PR_iter(s, a)))


class ActionValues(Values):
    def optim_action(self, s):
        if s is self.env.terminal:
            raise ValuesException('No action is available from the terminal state.')
        return max(self.env.actions(s), key=lambda a: self.v(s, a))

    def optim_value(self, s):
        if s is self.env.terminal:
            return 0
        return max(self.Q[s, a] for a in self.env.actions(s))

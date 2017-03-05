class statevalues(object):
    def __init__(self, env, initv=0):
        initf = initv if callable(initv) else lambda s: initv

        self.env = env
        self.V = {s: initf(s) for s in env.states()}
        self.V.update({s: 0 for s in env.states(begin=False, middle=False, terminal=True)})

    def __getitem__(self, s):
        return self.V[s]

    def __setitem__(self, s, v):
        self.V[s] = v

    def max_action(self, s0, model, with_argmax=False):
        maxv, maxa = max(((sum(p * self[s1] for p, s1, _ in model.PR_iter(s, a)), a) for a in self.actions(s0)), key=lambda va: v[0])
        return maxv, maxa if with_argmax else maxv

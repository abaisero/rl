class Environment:
    def __init__(self, mdp, objective):
        self.mdp = mdp
        self.objective = objective

    def reset(self):
        self.s = self.mdp.model.reset()
        self.g = 0
        self.t = 0
        self.obj = self.objective()
        self.obj.send(None)

    def step(self, a):
        self.s, r = self.mdp.model.step(self.s, a)
        self.g = self.obj.send(r)
        self.t += 1
        return r, self.s

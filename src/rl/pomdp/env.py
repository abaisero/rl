class Environment:
    def __init__(self, pomdp, objective):
        self.pomdp = pomdp
        self.objective = objective

    def reset(self):
        self.s = self.pomdp.model.reset()
        self.g = 0
        self.t = 0
        self.obj = self.objective()
        self.obj.send(None)

    def step(self, a):
        self.s, r, o = self.pomdp.model.step(self.s, a)
        self.g = self.obj.send(r)
        self.t += 1
        return r, o

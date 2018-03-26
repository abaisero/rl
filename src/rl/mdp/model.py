class Model:
    def __init__(self, env, s0model, s1model, rmodel):
        self.env = env
        self.s0 = s0model
        self.s1 = s1model
        self.r = rmodel

    def reset(self):
        s, = self.s0.sample()
        return s

    def step(self, s, a):
        s1, = self.s1.sample(s, a)
        r = self.r.sample(s, a, s1)
        return s1, r

class Model:
    def __init__(self, env, s0model, s1model, omodel, rmodel):
        self.env = env
        self.s0 = s0model
        self.s1 = s1model
        self.o = omodel
        self.r = rmodel
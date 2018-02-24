class StepSize:
    def __init__(self, init_s):
        self.init_s = init_s
        self.s = init_s

    def reset(self):
        self.s = self.init_s

    def __call__(self):
        return self.s

    def step(self):
        pass

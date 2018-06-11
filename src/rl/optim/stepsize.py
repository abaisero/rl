class StepSize:
    def __init__(self, init_s, decay=None):
        self.init_s = init_s
        self.decay = decay

        self.reset()

    def reset(self):
        self.s = self.init_s

    def __call__(self):
        return self.s

    def step(self):
        if self.decay is not None:
            self.s *= self.decay

from .stepsize import StepSize

class Geometric(StepSize):
    def __init__(self, init_s, decay):
        super().__init__(init_s)
        self.decay = decay

    def step(self):
        self.s *= self.decay

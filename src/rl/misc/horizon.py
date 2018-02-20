class Horizon:
    def __init__(self, H=None):
        self.H = H

    def __call__(self, t):
        return self.H is not None and t >= self.H

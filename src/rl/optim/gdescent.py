import numpy as np


class GDescent:
    def __init__(self, stepsize, clip=None):
        self.stepsize = stepsize
        self.clip = clip
        self.clip2 = clip ** 2 if clip is not None else None

    def __call__(self, grads):
        if self.clip is not None:
            gnorm2 = sum(g_.sum() for g_ in np.square(grads).flat)

            if self.clip2 < gnorm2:
                # NOTE avoiding overwrite
                c = self.clip / np.sqrt(gnorm2)
                grads = grads * c

        return self.stepsize * grads

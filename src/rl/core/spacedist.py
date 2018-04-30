import logging

import numpy as np
import numpy.random as rnd


class SpaceDistribution:
    logger = logging.getLogger(f'{__name__}.SpaceDistribution')

    def __init__(self, xspaces, yspaces, probs):
        self.xspaces = xspaces
        self.yspaces = yspaces
        self.probs = probs

        self.xyspaces = self.xspaces + self.yspaces
        self.nx = len(xspaces)
        self.ny = len(yspaces)
        self.nxy = len(self.xyspaces)

        self.ydims = tuple(s.nelems for s in yspaces)

    def pr(self, *xys):
        self.logger.debug(f'pr() \t; xy={xys}')
        assert len(xys) == self.nxy

        return self.probs[xys]

    def sample(self, *xs):
        self.logger.debug(f'sample() \t; x={xs}')
        assert len(xs) == self.nx

        probs = self.probs[xs].ravel()
        yi = rnd.multinomial(1, probs).argmax()
        yidxs = np.unravel_index(yi, self.ydims)
        ys = tuple(s.elem(i) for s, i in zip(self.yspaces, yidxs))
        return ys

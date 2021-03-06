import numpy as np


class Model:
    def __init__(self, *yspaces, cond=None, mask=None):
        xspaces = () if cond is None else tuple(cond)

        self.xspaces = xspaces
        self.yspaces = yspaces
        self.spaces = xspaces + yspaces

        self.xrank = len(self.xspaces)
        self.yrank = len(self.yspaces)
        self.rank = len(self.spaces)

        self.xdims = tuple(map(len, self.xspaces))
        self.ydims = tuple(map(len, self.yspaces))
        self.dims = self.xdims + self.ydims

        if mask is None:
            mask = np.ones(self.dims, dtype=np.bool)

        self.mask = mask

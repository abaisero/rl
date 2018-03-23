from .model import Model

import numpy as np
import numpy.random as rnd

import pytk.probability as probability


class Tabular(Model):
    def __init__(self, *yspaces, cond=None):
        super().__init__(*yspaces, cond=cond)
        # NOTE np.prod would return float 1.0 if xshape is empty
        self.xsize = np.prod(self.xdims, dtype=np.int64)
        self.ysize = np.prod(self.ydims)
        self.size = self.xsize * self.ysize

        self.xaxes = tuple(range(self.xrank))
        self.yaxes = tuple(range(self.xrank, self.rank))

        self.reset()

    def reset(self):
        self.params = np.ones(self.dims)
        self[:] = 1

    def __getitem__(self, elems):
        idxs = self.indices(elems)
        return self.params[idxs]

    def __setitem__(self, elems, value):
        idxs = self.indices(elems)
        self.params[idxs] = value
        params_normal = probability.normal(self.params)
        params_conditional = probability.conditional(params_normal, axis=self.xaxes)
        self.params = params_conditional

    @staticmethod
    def index(elem):
        try:
            return elem.idx
        except AttributeError:
            pass

        if elem is None: return slice(None)
        # if elem is Ellipsis: return slice(None)
        if isinstance(elem, slice): return elem

        assert False, 'Type of `elem` unknown?'

    def indices(self, elems, split=False):
        try:
            nelems = len(elems)
        except TypeError:
            elems = elems,
            nelems = 1
        elems += (slice(None),) * (self.rank - nelems)
        indices = tuple(self.index(elem) for elem in elems)
        if split:
            return indices[:self.xrank], indices[self.xrank:]
        return indices

    def logprobs(self, *elems):
        return np.log(self.probs(*elems))

    def probs(self, *elems):
        idxs = self.indices(elems)
        return self.params[idxs]

    #

    def dist(self, *xelems):
        assert len(xelems) == self.xrank

        probs = self.probs(*xelems)
        for yi in range(self.ysize):
            yidxs = np.unravel_index(yi, self.ydims)
            yelems = tuple(f.elem(i) for f, i in zip(self.yspaces, yidxs))
            yield yelems + (probs[yidxs],)

    def pr(self, *elems):
        assert len(elems) == self.rank

        return self.probs(*elems)

    def sample(self, *xelems):
        assert len(xelems) == self.xrank

        # TODO kinda like a JointFactory but without names;  just indices?

        probs = self.probs(*xelems).ravel()
        # yi = rnd.choice(self.ysize, p=probs)
        yi = rnd.multinomial(1, probs).argmax()
        yidxs = np.unravel_index(yi, self.ydims)
        yelems = tuple(f.elem(i) for f, i in zip(self.yspaces, yidxs))

        if len(yelems) == 1:
            return yelems[0]
        return yelems

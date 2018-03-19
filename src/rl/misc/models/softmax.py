from .model import Model

import numpy as np
import numpy.random as rnd
from scipy.misc import logsumexp

import pytk.probability as probability

# import string


class Softmax(Model):
    def __init__(self, *yspaces, cond=None):
        super().__init__(*yspaces, cond=cond)
        # NOTE np.prod would return float 1.0 if xshape is empty
        self.xsize = np.prod(self.xdims, dtype=np.int64)
        self.ysize = np.prod(self.ydims)
        self.size = self.xsize * self.ysize

        self.xaxes = tuple(range(self.xrank))
        self.yaxes = tuple(range(self.xrank, self.rank))

        ## subscripts for np.einsum
        # self.yss = string.ascii_lowercase[:self.yrank]
        # self.xss = string.ascii_lowercase[self.yrank:self.rank]
        # self.ss = self.yss + self.xss

        # precomputed features (done once)
        self.__phi = np.eye(self.size).reshape(2 * self.dims)

        self.reset()

    def reset(self):
        # self.params = np.zeros(self.dims)
        self.params = rnd.normal(size=self.dims)
        # self.params = 2 * rnd.normal(size=self.dims)
        # self.params = 3 * (.5 - rnd.random_sample(self.dims))

    @staticmethod
    def index(elem):
        try:
            return elem.idx
        except AttributeError:
            pass

        if elem is None: return slice(None)
        if elem is Ellipsis: return slice(None)
        # if isinstance(elem, slice): return elem

        raise Exception('What have you done')

    def indices(self, *elems, split=False):
        elems += (...,) * (self.rank - len(elems))
        indices = tuple(self.index(elem) for elem in elems)
        if split:
            return indices[:self.xrank], indices[self.xrank:]
        return indices

    def phi(self, *elems):
        idxs = self.indices(*elems)
        return self.__phi[idxs]

    def prefs(self, *elems):
        idxs = self.indices(*elems)
        return self.params[idxs]

    def logprobs(self, *elems, normalized=True):
        idxs = self.indices(*elems)

        prefs = self.params
        logprobs = prefs
        if normalized:
            logprobs -= logsumexp(prefs, axis=self.yaxes, keepdims=True)

        return logprobs[idxs]

    def probs(self, *elems, normalized=True):
        idxs = self.indices(*elems)

        logprobs = self.logprobs(normalized=False)
        probs = np.exp(logprobs - logprobs.max())
        if normalized:
            probs /= probs.sum(axis=self.yaxes, keepdims=True)

        return probs[idxs]

    def dprefs(self, *elems):
        return self.phi(*elems)

    def dlogprobs(self, *elems):
        idxs = self.indices(*elems)
        xidxs, yidxs = idxs[:self.xrank], idxs[self.xrank:]
        xungiven = sum(isinstance(idx, slice) for idx in xidxs)

        xelems = elems[:self.xrank]
        dprefs = self.dprefs(*xelems)
        probs = self.probs(*xelems)

        axes = tuple(range(xungiven, xungiven+self.yrank))
        dlogprobs = dprefs - np.tensordot(probs, dprefs, axes=(axes, axes))

        return dlogprobs[yidxs]

    def dprobs(self, *elems):
        idxs = self.indices(*elems)
        unindexed = sum(isinstance(idx, slice) for idx in idxs)

        probs = self.probs(*elems)
        dlogprobs = self.dlogprobs(*elems)
        dprobs = np.tensordot(probs, dlogprobs, axes=unindexed)
        return dprobs

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
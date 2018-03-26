from .model import Model

import numpy as np
import numpy.random as rnd
from scipy.special import logsumexp

import pytk.probability as probability

import string


class Softmax(Model):
    def __init__(self, *yspaces, cond=None):
        super().__init__(*yspaces, cond=cond)
        # NOTE np.prod would return float 1.0 if xshape is empty
        self.xsize = np.prod(self.xdims, dtype=np.int64)
        self.ysize = np.prod(self.ydims)
        self.size = self.xsize * self.ysize

        self.xaxes = tuple(range(self.xrank))
        self.yaxes = tuple(range(self.xrank, self.rank))

        # subscripts for np.einsum
        self.xss = string.ascii_lowercase[:self.xrank]
        self.yss = string.ascii_lowercase[self.xrank:self.rank]
        self.ss = self.xss + self.yss

        # precomputed features (done once)
        self.__phi = np.eye(self.size).reshape(2 * self.dims)

        self.reset()

    def reset(self):
        # self.params = np.zeros(self.dims)
        self.params = rnd.normal(size=self.dims)
        # self.params = 2 * rnd.normal(size=self.dims)
        # self.params = 3 * (.5 - rnd.random_sample(self.dims))

    def phi(self, *elems):
        return self.__phi[elems]

    def prefs(self, *elems):
        return self.params[elems]

    def logprobs(self, *elems, normalized=True):
        logprobs = prefs = self.params
        if normalized:
            logprobs -= logsumexp(prefs, axis=self.yaxes, keepdims=True)

        return logprobs[elems]

    def probs(self, *elems, normalized=True):
        logprobs = self.logprobs(normalized=False)
        probs = np.exp(logprobs - logprobs.max())
        if normalized:
            probs /= probs.sum(axis=self.yaxes, keepdims=True)

        return probs[elems]

    def dprefs(self, *elems):
        return self.phi(*elems)

    def dlogprobs(self, *elems):
        # TODO improve performance
        xelems = elems[:self.xrank]
        dprefs = self.dprefs()
        probs = self.probs()

        reshapekey = (slice(None),) * self.xrank + (np.newaxis,) * self.yrank + (...,)
        ss = f'{self.ss},{self.ss}...->{self.xss}...'
        dlogprobs = dprefs - np.einsum(ss, probs, dprefs)[reshapekey]
        return dlogprobs[elems]

    def dprobs(self, *elems):
        probs = self.probs(*elems)
        dlogprobs = self.dlogprobs(*elems)
        broadcast = (...,) + (np.newaxis,)*self.rank
        return probs[broadcast] * dlogprobs

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

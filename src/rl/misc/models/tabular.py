from .model import Model

import numpy as np
import numpy.random as rnd

import pytk.probability as probability


class Tabular(Model):
    def __init__(self, xdims, ydims, probs):
        self.probs = probs
        self.xdims = xdims
        self.ydims = ydims
        self.dims = xdims + ydims

        self.xrank = len(xdims)
        self.yrank = len(ydims)
        self.rank = self.xrank + self.yrank

        # NOTE np.prod would return float 1.0 if xshape is empty
        self.xsize = np.prod(self.xdims, dtype=np.int64)
        self.ysize = np.prod(self.ydims)
        self.size = self.xsize * self.ysize

        self.xaxes = tuple(range(self.xrank))
        self.yaxes = tuple(range(self.xrank, self.rank))

        self.xss = string.ascii_lowercase[:self.xrank]
        self.yss = string.ascii_lowercase[self.xrank:self.rank]
        self.ss = self.xss + self.yss

    def logprobs(self, elems):
        logprobs = params - logsumexp(params, axis=self.yaxes, keepdims=True)
        return logprobs[elems]

    def probs(self, elems):
        logprobs = self.logprobs(params, elems)
        return np.exp(logprobs)

    def phi(self, params, elems):
        return self.__phi[elems]

    def dprefs(self, params, elems):
        return self.phi(params, elems)

    def dlogprobs(self, params, elems):
        # TODO improve performance
        xelems = elems[:self.xrank]
        # dprefs = self.dprefs(params, elems)
        # probs = self.probs(params, elems)
        dprefs = self.dprefs(params, ())
        probs = self.probs(params, ())

        reshapekey = (slice(None),) * self.xrank + (np.newaxis,) * self.yrank + (...,)
        ss = f'{self.ss},{self.ss}...->{self.xss}...'
        dlogprobs = dprefs - np.einsum(ss, probs, dprefs)[reshapekey]
        return dlogprobs[elems]

    def dprobs(self, params, elems):
        probs = self.probs(params, elems)
        dlogprobs = self.dlogprobs(params, elems)
        broadcast = (...,) + self.rank * (np.newaxis,)
        return probs[broadcast] * dlogprobs

    def sample(self, params, xelems):
        if len(xelems) != self.xrank:
            raise ValueError(f'{self.xrank} inputs should be given;  {len(xelems)} given instead!')

        probs = self.probs(params, xelems).ravel()
        # yi = rnd.choice(self.ysize, p=probs)
        yi = rnd.multinomial(1, probs).argmax()
        ys = np.unravel_index(yi, self.ydims)
        return ys
        # return ys[0] if len(ys) == 1 else ys
        # yidxs = np.unravel_index(yi, self.ydims)
        # yelems = tuple(f.elem(i) for f, i in zip(self.yspaces, yidxs))

        # if len(yelems) == 1:
        #     return yelems[0]
        # return yelems

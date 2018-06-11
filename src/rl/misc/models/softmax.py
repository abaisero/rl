from .model import Model

import numpy as np
import numpy.random as rnd
from scipy.special import logsumexp

# import string


class Softmax(Model):
    def __init__(self, xdims, ydims, mask=None):
        self.mask = mask
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

        # self.xss = string.ascii_lowercase[:self.xrank]
        # self.yss = string.ascii_lowercase[self.xrank:self.rank]
        # self.ss = self.xss + self.yss

        self.__phi = np.eye(self.size).reshape(2 * self.dims)

    def new_params(self):
        params = .5 * rnd.randn(*self.dims)
        # params = np.zeros(self.dims)
        try:
            nmask = ~self.mask
        except TypeError:
            pass
        else:
            params[nmask] = -np.inf

        return self.process_params(params)

    def process_params(self, params, *, inline=False):
        if not inline:
            params = params.copy()
        params -= logsumexp(params, axis=self.yaxes, keepdims=True)
        return params

    def prefs(self, params, elems):
        return params[elems]

    def logprobs(self, params, elems):
        # NOTE not necessary anymore, with process_params
        # logprobs = params - logsumexp(params, axis=self.yaxes, keepdims=True)
        logprobs = params
        return logprobs[elems]

    def probs(self, params, elems):
        logprobs = self.logprobs(params, elems)
        return np.exp(logprobs)

    def phi(self, params, elems):
        return self.__phi[elems]

    def _dprefs(self, params, elems):
        return self.phi(params, elems)

    # NOTE naive implementation;  obsolete
    # def dlogprobs(self, params, elems):
    #     # TODO improve performance
    #     dprefs = self._dprefs(params, ())
    #     probs = self.probs(params, ())

    #     reshapekey = (
    #         (slice(None),) * self.xrank +
    #         (np.newaxis,) * self.yrank +
    #         (...,)
    #     )

    #     ss = f'{self.ss},{self.ss}...->{self.xss}...'
    #     dlogprobs = dprefs - np.einsum(ss, probs, dprefs)[reshapekey]
    #     return dlogprobs[elems]

    def dlogprobs(self, params, elems):
        xelems, yelems = elems[:self.xrank], elems[self.xrank:]
        dprefs = self._dprefs(params, xelems)
        probs = self.probs(params, xelems)

        yaxes = tuple(range(probs.ndim - self.yrank, probs.ndim))
        dlogprobs = dprefs - np.sum(probs * dprefs, axis=yaxes, keepdims=True)
        return dlogprobs[yelems]

    def dprobs(self, params, elems):
        probs = self.probs(params, elems)
        dlogprobs = self.dlogprobs(params, elems)
        broadcast = (...,) + self.rank * (np.newaxis,)
        return probs[broadcast] * dlogprobs

    def sample(self, params, xelems):
        if len(xelems) != self.xrank:
            raise ValueError(f'{self.xrank} inputs should be given;  '
                             f'{len(xelems)} given instead!')

        probs = self.probs(params, xelems).ravel()
        yi = rnd.multinomial(1, probs).argmax()
        ys = np.unravel_index(yi, self.ydims)
        return ys

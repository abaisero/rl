import logging
logger = logging.getLogger(__name__)

from .dist import Distribution

from pytk.decorators import nevernest, NestingError

import itertools as itt

import numpy as np
import numpy.random as rnd


class ScalarDistribution(Distribution):
    logger = logging.getLogger(f'{__name__}.ScalarDistribution')

    def __init__(self, *, cond=None):
        super().__init__(cond=cond)
        self.__array = None

    @property
    def array(self):
        if self.__array is None:
            shape = tuple(f.nelems for f in self.xspaces)
            self.__array = np.zeros(shape)
            for x in itt.product(*self.xspaces):
                xi = tuple(elem.idx for elem in x)
                self.__array[xi] = self.E(*x)

        return self.__array

    @array.setter
    def array(self, value):
        assert value.shape == tuple(f.nelems for f in self.xspaces)
        self.__array = value

    # TODO probably better wait..
    @nevernest(n=1)
    def dist(self, *x):
        logger.debug(f'dist() \t; x={x}')
        assert len(x) == self.nx

        if self.__array is not None:
            idx = tuple(elem.idx for elem in x)
            yield self.array[idx], 1.
            return

        raise NotImplementedError('Method self.dist() of this distribution was neither supplied nor can it be computed automagically.')

    # TODO probably better wait..
    @nevernest(n=1)
    def sample(self, *x):
        logger.debug(f'sample() \t; x={x}')
        assert len(x) == self.nx

        try:
            dist = list(self.dist(*x))
            rs = [r for r, _ in dist]
            ps = [p for _, p in dist]
            ri = rnd.multinomial(1, ps).argmax()
            r = rs[ri]
            # r = rnd.choice(rs, p=ps)
        except NestingError as e:
            raise NotImplementedError('Method self.sample() of this distribution was neither supplied nor can it be computed automagically.') from e

        return r

    # TODO probably better wait..
    @nevernest(n=1)
    def E(self, *x):
        logger.debug(f'E() \t; x={x}')
        assert len(x) == self.nx

        try:
            dist = self.dist(*x)
            E = sum(r * p for r, p in dist)
        except NestingError as e:
            raise NotImplementedError('Method self.E() of this distribution was neither supplied nor can it be computed automagically.') from e

        return E

from .fmodel import FModel

import numpy as np


class Tabular(FModel):
    def __init__(self, factories, initvalue):
        super().__init__(factories)
        self.initvalue = initvalue
        self.shape = tuple(f.nitems for f in factories)
        self.values = np.full(self.shape, initvalue)

    def reset(self, items=None):
        ind = self.index(items)
        self.values[ind] = self.initvalue

    def index(self, items):
        if isinstance(items, tuple):
            return tuple(
                item.i if factory.isitem(item) else item
                for factory, item in zip(self.factories, items)
            )

        factory, item = self.factories[0], items
        return item.i if factory.isitem(item) else item

    def __getitem__(self, items):
        ind = self.index(items)
        return self.values[ind]

    def __setitem__(self, items, value):
        ind = self.index(items)
        self.values[ind] = value

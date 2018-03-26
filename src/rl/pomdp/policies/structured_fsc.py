import logging
logger = logging.getLogger(__name__)

from .policy import Policy
import rl.graph as graph

import argparse

import indextools
import rl.misc.models as models

from types import SimpleNamespace

import numpy as np


class StructuredFSC(Policy):
    logger = logging.getLogger(f'{__name__}.StructuredFSC')

    def __init__(self, pomdp, nspace, n0, amask, nmask):
        super().__init__(pomdp)

        if not amask.shape[0] == pomdp.nactions:
            raise ValueError(f'Action mask shape {amask.shape} is wrong.')
        if not amask.shape[1] == nmask.shape[0] == nmask.shape[1]:
            raise ValueError(f'Action mask shape {amask.shape} and/or node mask shape {nmask.shape} is wrong.')

        self.nspace = nspace
        self.n0 = nspace.elem(n0)

        self.amask = amask
        self.nmask = nmask

        self.amodel = models.Softmax(pomdp.aspace, cond=(self.nspace,))
        self.nmodel = models.Softmax(self.nspace, cond=(self.nspace, pomdp.ospace))
        # TODO sparsity directly in model?

    def __repr__(self):
        return f'FSC_Structured(N={self.nnodes})'

    @property
    def params(self):
        # TODO need better way to handle multiparametric models...
        # maybe just concatenate?  seems wrong..
        params = np.empty(2, dtype=object)
        params[:] = self.amodel.params, self.nmodel.params
        return params

    @params.setter
    def params(self, value):
        aparams, nparams = value
        self.amodel.params = aparams
        self.nmodel.params = nparams

    def dlogprobs(self, n, a, o, n1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(n, a)
        dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
        return dlogprobs

    def new_pcontext(self):
        n = self.n0
        return SimpleNamespace(n=n)

    def reset(self):
        # TODO specific to softmax model;  there must be a better way
        self.amodel.reset()
        self.nmodel.reset()
        self.amodel.params[~self.amask.T] = -np.inf
        nobs = self.nmodel.params.shape[1]
        nmask = np.stack([self.nmask] * nobs, axis=1)
        self.nmodel.params[~nmask.T] = -np.inf

    @property
    def nodes(self):
        return self.nspace.elems

    @property
    def nnodes(self):
        return self.nspace.nelems

    def dist(self, pcontext):
        return self.amodel.dist(pcontext.n)

    def pr(self, pcontext, a):
        return self.amodel.pr(pcontext.n, a)

    def sample(self, pcontext):
        return self.amodel.sample(pcontext.n)

    def dist_n(self, n, o):
        return self.nmodel.dist(n, o)

    def pr_n(self, n, o, n1):
        return self.nmodel.pr(n, o, n1)

    def sample_n(self, n, o):
        return self.nmodel.sample(n, o)

    def plot(self, pomdp, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.structuredfscplot(self, pomdp, nepisodes)
        self.idx = 0

    def plot_update(self):
        adist = self.amodel.probs()
        adist /= adist.sum(axis=-1, keepdims=True)

        ndist = self.nmodel.probs()
        ndist /= ndist.sum(axis=-1, keepdims=True)

        self.q.put((self.idx, adist, ndist))
        self.idx += 1

        if self.idx == self.neps:
            self.q.put(None)

    @staticmethod
    def from_dotfss(self):
        pass


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('fss', type=str)

    parser.add_argument('--belief', action='store_const', const=True,
            default=False)

    @staticmethod
    def from_fss(pomdp, fname):
        with _open(fname) as f:
            dotfss = parse(f.read())

        # TODO check that all actions are used
        # TODO check that structure fully connected
        # TODO check that the actions are the same

        if isinstance(dotfss.nodes[0], str):
            nodes = dotfss.nodes
        else:
            nodes = tuple(f'node_{n}' for n in dotfss.nodes)

        nspace = indextools.DomainSpace(nodes)
        amask = dotfss.A.T
        nmask = dotfss.N.T

        return StructuredFSC(pomdp, nspace, dotfss.start, amask, nmask)

    @staticmethod
    def from_namespace(pomdp, namespace):
        return StructuredFSC.from_fss(pomdp, namespace.fss)




from contextlib import contextmanager
from pkg_resources import resource_filename

from rl_parsers.fss import parse


@contextmanager
def _open(fname):
    try:
        f = open(fname)
    except FileNotFoundError:
        fname = resource_filename('rl', f'data/fss/{fname}')
        f = open(fname)

    yield f
    f.close()

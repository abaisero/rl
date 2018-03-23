import logging
logger = logging.getLogger(__name__)

from .policy import Policy
import rl.graph as graph

import argparse
from rl.misc.argparse import GroupedAction

import indextools
import rl.misc.models as models

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import numpy.linalg as la
import numpy.random as rnd


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')


class StructuredFSC(Policy):
    logger = logging.getLogger(f'{__name__}.StructuredFSC')

    def __init__(self, env, nspace, n0, amask, nmask):
        super().__init__(env)

        if not amask.shape[0] == env.nactions:
            raise ValueError(f'Action mask shape {amask.shape} is wrong.')
        if not amask.shape[1] == nmask.shape[0] == nmask.shape[1]:
            raise ValueError(f'Action mask shape {amask.shape} and/or node mask shape {nmask.shape} is wrong.')

        self.nspace = nspace
        self.n0 = nspace.elem(n0)

        self.amask = amask
        self.nmask = nmask

        self.amodel = models.Softmax(env.aspace, cond=(self.nspace,))
        self.nmodel = models.Softmax(self.nspace, cond=(self.nspace, env.ospace))
        # TODO I think I want this sparsity to happen directly in a sparse fmodel

        # TODO look at pgradient;  this won't work for some reason
        # self.params = np.array([self.amodel.params, self.nmodel.params])

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
        # print(dlogprobs[0].shape)
        dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
        return dlogprobs

    def new_pcontext(self):
        n = self.n0
        return SimpleNamespace(n=n)

    def reset(self):
        self.amodel.reset()
        self.nmodel.reset()
        self.amodel.params[~self.amask.T] = -np.inf
        nmask = np.stack([self.nmask] * self.env.nobs, axis=1)
        self.nmodel.params[~nmask.T] = -np.inf

    # def restart(self):
    #     pass
        # self.n = self.n0.copy()

    @property
    def nodes(self):
        return self.nspace.elems

    @property
    def nnodes(self):
        return self.nspace.nelems

    @property
    def context(self):
        pass
        # return IContext(self.n)

    def feedback(self, feedback):
        pass
        # ifeedback = self.feedback_o(feedback.o)
        # self.logger.debug(f'feedback({feedback}) -> {ifeedback}')
        # return ifeedback

    def feedback_o(self, o):
        pass
        # n1 = self.nmodel.sample(self.n, o)
        # ifeedback = IFeedback(n1=n1)
        # self.logger.debug(f'feedback_o(o) + {self.n} -> {ifeedback}')

        # self.n = n1
        # return ifeedback

    def dist(self, pcontext):
        # return self.amodel.dist(self.n)
        return self.amodel.dist(pcontext.n)

    def pr(self, pcontext, a):
        # return self.amodel.pr(self.n, a)
        return self.amodel.pr(pcontext.n, a)

    def sample(self, pcontext):
        # return self.amodel.sample(self.n)
        return self.amodel.sample(pcontext.n)

    def sample_n(self, n, o):
        return self.nmodel.sample(n, o)

    def plot(self, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.structuredfscplot(self, nepisodes)
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

    # @staticmethod
    # def parser(group=None):
    #     def group_fmt(dest):
    #         return dest if group is None else f'{group}.{dest}'

    #     parser = argparse.ArgumentParser(add_help=False)
    #     parser.add_argument(dest=group_fmt('fss'), metavar='fss',
    #             action=GroupedAction, default=argparse.SUPPRESS)

    #     parser.add_argument('--belief', action='store_const', const=True,
    #             default=False)

    #     return parser

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('fss', type=str)

    parser.add_argument('--belief', action='store_const', const=True,
            default=False)


    @staticmethod
    def from_fss(env, fname):
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

        return StructuredFSC(env, nspace, dotfss.start, amask, nmask)

    @staticmethod
    def from_namespace(env, namespace):
        return StructuredFSC.from_fss(env, namespace.fss)




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

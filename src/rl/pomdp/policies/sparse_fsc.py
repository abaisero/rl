from .policy import Policy
import rl.graph as graph

import argparse
from rl.misc.argparse import GroupedAction

import indextools
import rl.misc.models as models


from collections import namedtuple
import numpy as np
import numpy.linalg as la
import numpy.random as rnd


IContext = namedtuple('IContext', 'n')
IFeedback = namedtuple('IFeedback', 'n1')


class SparseFSC(Policy):
    def __init__(self, env, N, K):
        super().__init__(env)
        self.N = N  # number of nodes
        self.K = K

        nodes = [f'node_{i}' for i in range(N)]
        # self.nspace = indextools.RangeSpace(K)
        self.nspace = indextools.DomainSpace(nodes)
        self.kspace = indextools.RangeSpace(K)

        # TODO first node should probably not be sparse..

        self.amodel = models.Softmax(env.aspace, cond=(self.nspace,))
        self.kmodel = models.Softmax(self.kspace, cond=(self.nspace, env.ospace))

        I = np.eye(N, dtype=np.int)
        fail = 0
        while True:
            cols = (rnd.choice(N, K, replace=False) for _ in range(N))
            nkn = (I[:, col] for col in cols)
            nkn = np.stack(nkn, axis=-1)
            nn = nkn.sum(axis=1)

            test = la.multi_dot([nn] * N)
            if np.all(test > 0):
                break
            else:
                if fail == 100:
                    raise Exception
                fail += 1
        self.nkn = nkn
        self.nn = nn

        # TODO look at pgradient;  this won't work for some reason
        # self.params = np.array([self.amodel.params, self.kmodel.params])

    @property
    def params(self):
        # TODO need better way to handle multiparametric models...
        # maybe just concatenate?  seems wrong..
        params = np.empty(2, dtype=object)
        params[:] = self.amodel.params, self.kmodel.params
        return params

    @params.setter
    def params(self, value):
        aparams, oparams = value
        self.amodel.params = aparams
        self.kmodel.params = oparams

    def nk2n(self, n, k):
        #  I thought it should have been [:, k.i, n.i]...
        n1i = self.nkn[:, k.i, n.i].nonzero()[0].item()
        return self.nspace.elem(n1i)

    def nn2k(self, n, n1):
        k1i = self.nkn[n1.i, :, n.i].nonzero()[0].item()
        return self.kspace.elem(k1i)

    def dlogprobs(self, n, a, o, n1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(n, a)
        k1 = self.nn2k(n, n1)
        dlogprobs[1] = self.kmodel.dlogprobs(n, o, k1)
        return dlogprobs

    def reset(self):
        self.amodel.reset()
        self.kmodel.reset()

    def restart(self):
        self.n = self.nspace.elem(0)

    @property
    def nodes(self):
        return self.nspace.nelems

    @property
    def nnodes(self):
        return self.nspace.nelems

    @property
    def context(self):
        return IContext(self.n)

    def feedback(self, feedback):
        return self.feedback_o(feedback.o)

    def feedback_o(self, o):
        k = self.kmodel.sample(self.n, o)
        self.n = self.nk2n(self.n, k)
        return IFeedback(n1=self.n)

    def dist(self):
        return self.amodel.dist(self.n)

    def pr(self, a):
        return self.amodel.pr(self.n, a)

    def sample(self):
        self.a = self.amodel.sample(self.n)
        return self.a

    def plot(self, nepisodes):
        self.neps = nepisodes
        self.q, self.p = graph.sparsefscplot(self, nepisodes)
        self.idx = 0

    def plot_update(self):
        adist = self.amodel.probs()
        # NOTE this was just for bug
        # adist /= adist.sum(axis=-1, keepdims=True)

        kdist = self.kmodel.probs()
        # NOTE this was just for bug
        # kdist /= kdist.sum(axis=-1, keepdims=True)
        ndist = np.einsum('nok,mkn->mon', kdist, self.nkn)

        self.q.put((self.idx, adist, ndist))
        self.idx += 1

        if self.idx == self.neps:
            self.q.put(None)

    @staticmethod
    def parser(group=None):
        def group_fmt(dest):
            return dest if group is None else f'{group}.{dest}'

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(dest=group_fmt('n'), metavar='n', type=int, action=GroupedAction, default=argparse.SUPPRESS)
        parser.add_argument(dest=group_fmt('k'), metavar='k', type=int, action=GroupedAction, default=argparse.SUPPRESS)
        return parser

    @staticmethod
    def from_namespace(env, namespace):
        return SparseFSC(env, namespace.n, namespace.k)

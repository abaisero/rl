from .policy import Policy
import rl.graph as graph

import pytk.factory as factory
import pytk.factory.model as fmodel

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

        values = [f'node_{i}' for i in range(N)]
        # self.nfactory = factory.FactoryN(N)
        self.nfactory = factory.FactoryValues(values)
        self.kfactory = factory.FactoryN(K)

        # TODO first node should probably not be sparse..

        self.amodel = fmodel.Softmax(env.afactory, cond=(self.nfactory,))
        self.kmodel = fmodel.Softmax(self.kfactory, cond=(self.nfactory, env.ofactory))
        # TODO I think I want this sparsity to happen directly in a sparse fmodel

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
        return self.nfactory.item(n1i)

    def nn2k(self, n, n1):
        k1i = self.nkn[n1.i, :, n.i].nonzero()[0].item()
        return self.kfactory.item(k1i)

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
        self.n = self.nfactory.item(0)

    @property
    def nodes(self):
        return self.nfactory.items

    @property
    def nnodes(self):
        return self.nfactory.nitems

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
        adist /= adist.sum(axis=-1, keepdims=True)

        kdist = self.kmodel.probs()
        kdist /= kdist.sum(axis=-1, keepdims=True)
        ndist = np.einsum('nok,mkn->nom', kdist, self.nkn)

        self.q.put((self.idx, adist, ndist))
        self.idx += 1

        if self.idx == self.neps:
            self.q.put(None)

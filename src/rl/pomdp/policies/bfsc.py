# from .policy import Policy
# import rl.graph as graph

# # import argparse
# # from rl.misc.argparse import GroupedAction

# import indextools
# import rl.misc.models as models

# from collections import namedtuple
# from types import SimpleNamespace

# import numpy as np


# IContext = namedtuple('IContext', 'n')
# IFeedback = namedtuple('IFeedback', 'n1')

# # PContext = SimpleNamespace('PContext', 'amodel, nmodel, n')


# class BeliefFSC(Policy):
#     def __init__(self, env, fsc):
#         # TODO belief over contexts?!
#         # TODO only makes sense for fscs with node context.... ok
#         super().__init__(env)
#         self.fsc = fsc

#     def __repr__(self):
#         return f'Belief-{self.fsc}'

#     @property
#     def params(self):
#         return self.fsc.params

#     @params.setter
#     def params(self, value):
#         self.fsc.params = value

#     def dlogprobs(self, n, a, o, n1):
#         dlogprobs = np.empty(2, dtype=object)
#         dlogprobs[0] = self.amodel.dlogprobs(n, a)
#         dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
#         return dlogprobs

#     def new_pcontext(self):
#         b = models.Tabular(self.fsc.nspace)
#         return SimpleNamespace(b=b)

#     def reset(self):
#         self.fsc.reset()

#     def restart(self):
#         pass

#     @property
#     def nspace(self):
#         return self.fsc.nspace

#     @property
#     def nodes(self):
#         return self.nspace.elems

#     @property
#     def nnodes(self):
#         return self.nspace.nelems

#     # @property
#     # def context(self):
#     #     pass
#     #     # return IContext(self.n)

#     # def feedback(self, feedback):
#     #     pass
#     #     # return self.feedback_o(feedback.o)

#     # def feedback_o(self, o):
#     #     pass
#     #     # self.n = self.nmodel.sample(self.n, o)
#     #     # return IFeedback(n1=self.n)

#     @property
#     def amodel(self):
#         return self.fsc.amodel

#     @property
#     def nmodel(self):
#         return self.fsc.nmodel

#     def dist(self, pcontext):
#         raise NotImplementedError
#         # return self.amodel.dist(self.n)
#         # return pcontext.b. self.amodel.dist(pcontext.n)

#     def pr(self, pcontext, a):
#         raise NotImplementedError
#         # return self.amodel.pr(self.n, a)
#         return self.amodel.pr(pcontext.n, a)

#     def sample(self, pcontext):
#         pcontext.n = pcontext.b.sample()
#         return self.fsc.sample(pcontext)

#     # def sample_n(self, n, o):
#     #     return self.nmodel.sample(n, o)


#     def plot(self, pomdp, nepisodes):
#         self.fsc.plot(pomdp, nepisodes)
#         # raise NotImplementedError
#         # self.neps = nepisodes
#         # self.q, self.p = graph.fscplot(self, nepisodes)
#         # self.idx = 0

#     def plot_update(self):
#         self.fsc.plot_update()
#         # raise NotImplementedError
#         # adist = self.amodel.probs()
#         # adist /= adist.sum(axis=-1, keepdims=True)

#         # ndist = self.nmodel.probs()
#         # ndist /= ndist.sum(axis=-1, keepdims=True)

#         # self.q.put((self.idx, adist, ndist))
#         # self.idx += 1

#         # if self.idx == self.neps:
#         #     self.q.put(None)








import argparse
import copy
import types

import indextools
import rl.misc.models as models
import rl.graph as graph


import numpy as np


class BFSC:
    # TODO I want to /inherit/ from others...
    def __init__(self, fsc):
        self.fsc = fsc

    @property
    def nodes(self):
        return self.fsc.nodes

    @property
    def nnodes(self):
        return self.fsc.nnodes





    @property
    def params(self):
        return self.fsc.params

    @params.setter
    def params(self, value):
        self.fsc.params = value

    def dlogprobs(self, n, a, o, n1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(n, a)
        dlogprobs[1] = self.nmodel.dlogprobs(n, o, n1)
        return dlogprobs

    def new_pcontext(self):
        b = models.Tabular(self.fsc.nspace)
        return SimpleNamespace(b=b)

    def reset(self):
        self.fsc.reset()

    def restart(self):
        pass

    @property
    def nspace(self):
        return self.fsc.nspace

    @property
    def nodes(self):
        return self.nspace.elems

    @property
    def nnodes(self):
        return self.nspace.nelems

    # @property
    # def context(self):
    #     pass
    #     # return IContext(self.n)

    # def feedback(self, feedback):
    #     pass
    #     # return self.feedback_o(feedback.o)

    # def feedback_o(self, o):
    #     pass
    #     # self.n = self.nmodel.sample(self.n, o)
    #     # return IFeedback(n1=self.n)

    @property
    def amodel(self):
        return self.fsc.amodel

    @property
    def nmodel(self):
        return self.fsc.nmodel

    def dist(self, pcontext):
        raise NotImplementedError
        # return self.amodel.dist(self.n)
        # return pcontext.b. self.amodel.dist(pcontext.n)

    def pr(self, pcontext, a):
        raise NotImplementedError
        # return self.amodel.pr(self.n, a)
        return self.amodel.pr(pcontext.n, a)

    def sample(self, pcontext):
        pcontext.n = pcontext.b.sample()
        return self.fsc.sample(pcontext)

    # def sample_n(self, n, o):
    #     return self.nmodel.sample(n, o)


    def plot(self, pomdp, nepisodes):
        self.fsc.plot(pomdp, nepisodes)
        # raise NotImplementedError
        # self.neps = nepisodes
        # self.q, self.p = graph.fscplot(self, nepisodes)
        # self.idx = 0

    def plot_update(self):
        self.fsc.plot_update()
        # raise NotImplementedError
        # adist = self.amodel.probs()
        # adist /= adist.sum(axis=-1, keepdims=True)

        # ndist = self.nmodel.probs()
        # ndist /= ndist.sum(axis=-1, keepdims=True)

        # self.q.put((self.idx, adist, ndist))
        # self.idx += 1

        # if self.idx == self.neps:
        #     self.q.put(None)








































class BFSC(FSC):
    def new_context(self, params):
        bn = ...
        return types.SimpleNamespace(n=self.nspace.elem(0))

    def step(self, params, pcontext, feedback, *, inline=True):
        n1 = self.sample_n(params, pcontext, feedback)

        if not inline:
            pcontext = copy.copy(pcontext)
        pcontext.n = n1

        return pcontext

    def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(params[0], (pcontext.n, a))
        dlogprobs[1] = self.nmodel.dlogprobs(params[1], (pcontext.n, feedback.o, pcontext1.n))
        return dlogprobs

    def pr_a(self, params, pcontext):
        return self.amodel.pr(params[0], (pcontext.n,))

    def sample_a(self, params, pcontext):
        a, = self.amodel.sample(params[0], (pcontext.n,))
        return a

    def pr_n(self, params, pcontext, feedback):
        return self.nmodel.pr(params[1], (pcontext.n, feedback.o))

    def sample_n(self, params, pcontext, feedback):
        n, = self.nmodel.sample(params[1], (pcontext.n, feedback.o))
        return n




class BFSC:
    def __init__(self, fsc):
        self.fsc = fsc

    @property
    def nodes(self):
        return self.fsc.nodes

    @property
    def nnodes(self):
        return self.fsc.nnodes

    @property
    def amodel(self):
        return self.fsc.amodel

    @property
    def nmodel(self):
        return self.fsc.nmodel

    def new_params(self):
        return self.fsc.new_params()

    def new_context(self, params):
        bn = np.ones(self.nnodes) / self.nnodes
        return types.SimpleNamespace(bn=bn0)

    def step(self, params, pcontext, feedback, *, inline=True):
        ...

    def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(params[0], (pcontext.n, a))
        dlogprobs[1] = self.nmodel.dlogprobs(params[1], (pcontext.n, feedback.o, pcontext1.n))
        return dlogprobs

    def pr_a(self, params, pcontext):
        return self.amodel.pr(params[0], (pcontext.n,))

    def sample_a(self, params, pcontext):
        n = self.sample_n(self, params, pcontext)
        a, = self.amodel.sample(params[0], (pcontext.n,))
        return a




    def new_context(self, params):
        bn = ...
        return types.SimpleNamespace(n=self.nspace.elem(0))

    def step(self, params, pcontext, feedback, *, inline=True):
        n1 = self.sample_n(params, pcontext, feedback)

        if not inline:
            pcontext = copy.copy(pcontext)
        pcontext.n = n1

        return pcontext

    def dlogprobs(self, params, pcontext, a, feedback, pcontext1):
        dlogprobs = np.empty(2, dtype=object)
        dlogprobs[0] = self.amodel.dlogprobs(params[0], (pcontext.n, a))
        dlogprobs[1] = self.nmodel.dlogprobs(params[1], (pcontext.n, feedback.o, pcontext1.n))
        return dlogprobs

    def pr_a(self, params, pcontext):
        return self.amodel.pr(params[0], (pcontext.n,))

    def sample_a(self, params, pcontext):
        n = sample_n(self, params, pcontext)
        a, = self.amodel.sample(params[0], (n,))
        return a

    def pr_n(self, params, pcontext):
        return self.nmodel.pr(params[1], (pcontext.n,))

    def sample_n(self, params, pcontext):
        n, = self.nmodel.sample(params[1], (pcontext.n,))
        return n

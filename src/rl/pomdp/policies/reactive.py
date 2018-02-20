from .policy import Policy

import pytk.factory.model as fmodel

from collections import namedtuple


IContext = namedtuple('IContext', 'o')
IFeedback = namedtuple('IFeedback', 'o1')

# TODO reset!!!!! how?! does one receive an observation to begin with?

class Reactive(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.amodel = fmodel.Softmax(env.afactory, cond=(env.ofactory,))

    def reset(self):
        self.amodel.reset()

    def restart(self):
        # TODO how to select first obs state?
        self.o = self.env.ofactory.item(0)

    @property
    def context(self):
        return IContext(self.o)

    def feedback(self, o):
        self.o = o
        return IFeedback(o1=o)  # TODO what is this for..

    def dist(self):
        return self.amodel.dist(self.o)

    def pr(self, a):
        return self.amodel.pr(self.o, a)

    def sample(self):
        return self.amodel.sample(self.o)


# TODO not quite reactive
# class ReactiveM(Policy):
#     def __init__(self, env, M):
#         super().__init__(env)
#         self.M = M

#         cond = M * (env.ofactory,)
#         self.amodel = fmodel.Softmax(env.afactory, cond=cond)


#     def reset(self):
#         self.os = deque([], self.M)

#     @property
#     def context(self):
#         return IContext(self.os)

#     def feedback(self, o):
#         pass

#     def dist(self):
#         return self.amodel.dist(*self.d)

#     def pr(self, a):
#         return self.amodel.pr(*self.d, a)

#     def sample(self):
#         return self.amodel.sample(*self.d)

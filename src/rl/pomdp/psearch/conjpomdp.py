import logging
logger = logging.getLogger(__name__)

from .p import P

import numpy as np


class CONJPOMDP(P):
    """ "Experiments with Infinite-Horizon, Policy-Gradient Estimation" - J. Baxter, P. Bartless, L. Weaver """

    logger = logging.getLogger(f'{__name__}.CONJPOMDP')

    def __init__(self, policy, grad, step_size, eps):
        super().__init__(policy)
        self.grad = grad
        self.step_size = step_size
        self.eps = eps

    def reset(self):
        self.grad.reset()
        self.pgen = None

    def restart(self):
        self.grad.restart()

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        return self.grad.feedback(sys, context, a, feedback)

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        # dparams = self.grad.feedback_episode(sys, episode)
        # try:
        #     pgen_send = self.pgen.send
        # except AttributeError:
        #     params = self.policy.amodel.params
        #     self.pgen = self.conjpomdp(params, dparams)
        #     return self.pgen.send(None)

        # return pgen_send(dparams)

        dparams = self.grad.feedback_episode(sys, episode)
        if self.pgen is None:
            params = self.policy.params
            self.pgen = self.conjpomdp(params, dparams)
            # TODO these stop iterations are confusing... and might be wrong..
            try:
                return next(self.pgen)
            except StopIteration:
                self.pgen = None
                return None
        try:
            return self.pgen.send(dparams)
        except StopIteration:
            self.pgen = None
            return None

    @staticmethod
    def inner(x, y):
        return sum(_.sum() for _ in x*y)

    def conjpomdp(self, params, dparams):
        g = h = dparams
        while self.inner(g, g) >= self.eps:
            params = yield from self.gsearch(params, h)
            delta = yield params
            gamma = self.inner(delta-g, delta) / self.inner(g, g)
            h = delta + gamma * h
            if self.inner(h, delta) < 0:
                h = delta
            g = delta

    def gsearch(self, params, dparams):
        s = self.step_size
        delta = yield params + s * dparams
        if self.inner(delta, dparams) < 0:
            # step back to bracket the maximum
            while True:
                sp = s
                pp = self.inner(delta, dparams)
                s /= 2
                delta = yield params + s * dparams

                # if np.inner(delta.reshape(-1), dparams.reshape(-1)) > -self.eps:
                if self.inner(delta, dparams) > -self.eps:
                    break
            sm = s
            pm = self.inner(delta, dparams)
        else:
            # step forward to bracket the maximum
            while True:
                sm = s
                pm = self.inner(delta, dparams)
                s *= 2
                delta = yield params + s * dparams

                if self.inner(delta, dparams) < self.eps:
                    break
            sp = s
            pp = self.inner(delta, dparams)

        if pm > 0 and pp < 0:
            s = sm - pm * (sp - sm) / (pp - pm)
        else:
            s = (sm + sp) / 2

        # self.step_size = s
        return params + s * dparams

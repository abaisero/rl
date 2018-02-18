import rl.pomdp as pomdp
import pytk.factory as factory

import numpy.random as rnd


class Tiger_S0Model(pomdp.State0Distribution):
    def __init__(self, env):
        super().__init__(env)
        self.ps = 1 / env.nstates

    def dist(self):
        for s in self.env.states:
            yield s, self.ps


class Tiger_S1Model(pomdp.State1Distribution):
    def dist(self, s0, a):
        if a == 'listen':
            yield s0.copy(), 1.
        else:
            p = 1 / self.env.nstates
            for s in self.env.states:
                yield s, p


class Tiger_OModel(pomdp.ObsDistribution):
    def __init__(self, env, e):
        super().__init__(env)
        self.e = e

    def dist(self, s0, a, s1):
        if a != 'listen':
            yield self.env.ofactory.item(value='none'), 1.
        else:
            if s0 == 'tiger-left':
                otrue, ofalse = 'hear-tiger-left', 'hear-tiger-right'
            else:
                otrue, ofalse = 'hear-tiger-right', 'hear-tiger-left'

            yield self.env.ofactory.item(value=otrue), 1-self.e
            yield self.env.ofactory.item(value=ofalse), self.e


class Tiger_RModel(pomdp.RewardDistribution):
    def dist(self, s0, a, s1):
        if a == 'listen':
            yield -1, 1.
        elif (a == 'open-left' and s0 == 'tiger-left') or \
            (a == 'open-right' and s0 == 'tiger-right'):
            yield 10, 1.
        else:
            yield -100, 1.



def Tiger(e=.2):

    svalues = 'tiger-left', 'tiger-right'
    sfactory = factory.FactoryValues(svalues)
    sfactory.istr = lambda item: f'State({item.value})'

    avalues = 'listen', 'open-left', 'open-right'
    afactory = factory.FactoryValues(avalues)
    afactory.istr = lambda item: f'Action({item.value})'

    ovalues = 'hear-tiger-left', 'hear-tiger-right', 'none'
    ofactory = factory.FactoryValues(ovalues)
    ofactory.istr = lambda item: f'Obs({item.value})'

    env = pomdp.Environment(sfactory, afactory, ofactory)

    s0model = Tiger_S0Model(env)
    s1model = Tiger_S1Model(env)
    omodel = Tiger_OModel(env, e)
    rmodel = Tiger_RModel(env)
    env.model = pomdp.Model(env, s0model, s1model, omodel, rmodel)

    return env

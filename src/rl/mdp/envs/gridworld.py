import rl.mdp as mdp
import indextools

import numpy as np
import numpy.random as rnd

from collections import namedtuple


n = 5
State = namedtuple('State', 'pos')

steps = dict(
    north = [-1,  0],
    south = [ 1,  0],
    east  = [ 0,  1],
    west  = [ 0, -1],
)

def T(s0, a):
    """ deterministic movement transition """
    s1pos = np.array(s0.pos) + steps[a]
    s1pos = tuple(s1pos.clip(0, n-1))
    return State(s1pos)


class Gridworld_S0Model(mdp.State0Distribution):
    def __init__(self, env):
        super().__init__(env)
        self.ps = 1 / env.nstates

    def dist(self):
        for s in self.env.states:
            yield s, self.ps


class Gridworld_S1Model(mdp.State1Distribution):
    def dist(self, s, a):
        if s == State((0, 1)):
            s1 = State((4, 1))
        elif s == State((0, 3)):
            s1 = State((2, 3))
        else:
            s1 = T(s.value, a.value)

        yield self.env.sspace.elem(value=s1), 1.
        # yield s1, 1.


class Gridworld_RModel(mdp.RewardDistribution):
    def dist(self, s, a, s1):
        if s == State((0, 1)):
            yield 10, 1.
        elif s == State((0, 3)):
            yield 5, 1.
        elif s == s1:
            yield -1, 1.
        else:
            yield 0, 1.


def Gridworld():
    # TODO avoid naive way!  or maybe not?  I want positions are tuples!
    svalues = [State((i, j)) for i in range(n) for j in range(n)]
    sspace = indextools.DomainSpace(svalues)
    # sspace.istr = lambda s: str(s.value.pos)
    # NOTE namedtuple takes care of map to string

    avalues = 'north', 'south', 'east', 'west'
    aspace = indextools.DomainSpace(avalues)
    # aspace.istr = lambda a: f'Action({a.value})'

    env = mdp.Environment(sspace, aspace)
    env.gamma = 1
    env.n = n

    s0model = Gridworld_S0Model(env)
    s1model = Gridworld_S1Model(env)
    rmodel = Gridworld_RModel(env)
    env.model = mdp.Model(env, s0model, s1model, rmodel)

    return env

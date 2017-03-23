from __future__ import division

import numpy as np
import numpy.random as rnd

from pytk.decorators import memoizemethod

from rl.env import environment, model, ModelException
from rl.values import actionvalues
from rl.algo.mc import on_policy_mc
from rl.policy import egreedy

import matplotlib.pyplot as plt


class racetrack_environment(environment):
    def __init__(self, track, maxvel):
        self.track = track
        self.vels = [(vi, vj) for vi in xrange(maxvel+1) for vj in xrange(maxvel+1) if vi != 0 or vj != 0]

        nrows, ncols = track.shape

        self.__states_begin = [(0, j, 1, 0) for j in xrange(ncols) if track[0, j]]
        self.__states_middle = [(i, j, vi, vj) for i in xrange(nrows) for j in xrange(ncols) for vi, vj in self.vels]
        self.__states_terminal = [(i, ncols-1, vi, vj) for i in xrange(nrows) if track[i, -1] for vi, vj in self.vels]

        self.__actions = [(dvi, dvj) for dvi in (-1, 0, 1) for dvj in (-1, 0, 1)]

    @memoizemethod
    def intrack(self, pos):
        try:
            return bool(track[pos])
        except IndexError:
            return False

    @memoizemethod
    def states(self, begin=True, middle=True, terminal=False):
        return begin * self.__states_begin \
                + middle * self.__states_middle \
                + terminal * self.__states_terminal

    @memoizemethod
    def actions(self, s):
        i, j, vi, vj = s
        return [(dvi, dvj) for dvi, dvj in self.__actions if (vi+dvi, vj+dvj) in self.vels]

class racetrack_model(model):
    @memoizemethod
    def sample_s1(self, a, s0):
        if a not in self.env.actions(s0):
            raise ModelException('Action not available')

        dvi, dvj = a
        i0, j0, vi0, vj0 = s0

        # which state does s0 + a amount to?
        p1 = i0 + vi0 + dvi, j0 + vj0 + dvj
        v1 =      vi0 + dvi,      vj0 + dvj
        s1 = p1 + v1

        if self.env.intrack(p1):
            return s1

        candidates = []

        # move horizontal
        p01 = i0 + 0, j0 + 1
        # v01 = vi0 + dvi, vj0 + dvj
        v01 = vi0, vj0
        s01 = p01 + v01

        if self.env.intrack(p01):
            candidates += [s01]

        # move vertical
        p01 = i0 + 1, j0 + 0
        # v01 = vi0 + dvi, vj0 + dvj
        v01 = vi0, vj0
        s01 = p01 + v01

        if self.env.intrack(p01):
            candidates += [s01]

        print s0, a, candidates
        i = rnd.choice(len(candidates))
        return candidates[i]

    @memoizemethod
    def R(self, a, s0, s1):
        if a not in self.env.actions(s0):
            return 0

        dvi, dvj = a
        i0, j0, vi0, vj0 = s0
        i1, j1, vi1, vj1 = s1

        p01 = i0 + vi0 + dvi, j0 + vj0 + dvj
        v01 = vi0 + dvi, vj0 + dvj
        s01 = p01 + v01

        return -1 if self.env.intrack(p01) else -5


track1 = np.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.bool)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3, linewidth=np.inf)

    track = track1
    maxvel = 4
    eps = .05

    env = racetrack_environment(track[::-1, :], maxvel)
    mod = racetrack_model(env)
    Q = actionvalues(env)
    policy = egreedy(env, eps)

    gamma = 1

    for s in env.states(begin=True, middle=False, terminal=False):
        print s, policy[s]

    op_mc = on_policy_mc(env, mod, policy, Q, gamma)
    op_mc.run()

    for s in env.states(begin=True, middle=False, terminal=False):
        print s, policy[s]

    
    plt.matshow(track)
    for i in xrange(1):
        episodes = op_mc.sim.run()
        for s0, _, _, s1 in episodes:
            i0, j0, _, _ = s0
            i1, j1, _, _ = s1
            plt.plot([i0, i1], [j0, j1])

    plt.show()

    # print policy

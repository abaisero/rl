# from __future__ import division

# import numpy as np
# import numpy.random as rnd

# from pytk.decorators import memoizemethod

# from rl.env import environment, model, ModelException
# from rl.values import actionvalues
# from rl.algo.mc import on_policy_mc
# from rl.policy import egreedy

# import matplotlib.pyplot as plt


# class racetrack_environment(environment):
#     def __init__(self, track, maxvel):
#         self.track = track
#         self.vels = [(vi, vj) for vi in xrange(maxvel+1) for vj in xrange(maxvel+1) if vi != 0 or vj != 0]

#         nrows, ncols = track.shape

#         self.__states_begin = [(0, j, 1, 0) for j in xrange(ncols) if track[0, j]]
#         self.__states_middle = [(i, j, vi, vj) for i in xrange(nrows) for j in xrange(ncols) for vi, vj in self.vels]
#         self.__states_terminal = [(i, ncols-1, vi, vj) for i in xrange(nrows) if track[i, -1] for vi, vj in self.vels]

#         self.__actions = [(dvi, dvj) for dvi in (-1, 0, 1) for dvj in (-1, 0, 1)]

#     @memoizemethod
#     def intrack(self, pos):
#         try:
#             return bool(track[pos])
#         except IndexError:
#             return False

#     @memoizemethod
#     def states(self, begin=True, middle=True, terminal=False):
#         return begin * self.__states_begin \
#                 + middle * self.__states_middle \
#                 + terminal * self.__states_terminal

#     @memoizemethod
#     def actions(self, s):
#         i, j, vi, vj = s
#         return [(dvi, dvj) for dvi, dvj in self.__actions if (vi+dvi, vj+dvj) in self.vels]

# class racetrack_model(model):
#     @memoizemethod
#     def sample_s1(self, a, s0):
#         if a not in self.env.actions(s0):
#             raise ModelException('Action not available')

#         dvi, dvj = a
#         i0, j0, vi0, vj0 = s0

#         # which state does s0 + a amount to?
#         p1 = i0 + vi0 + dvi, j0 + vj0 + dvj
#         v1 =      vi0 + dvi,      vj0 + dvj
#         s1 = p1 + v1

#         if self.env.intrack(p1):
#             return s1

#         candidates = []

#         # move horizontal
#         p01 = i0 + 0, j0 + 1
#         # v01 = vi0 + dvi, vj0 + dvj
#         v01 = vi0, vj0
#         s01 = p01 + v01

#         if self.env.intrack(p01):
#             candidates += [s01]

#         # move vertical
#         p01 = i0 + 1, j0 + 0
#         # v01 = vi0 + dvi, vj0 + dvj
#         v01 = vi0, vj0
#         s01 = p01 + v01

#         if self.env.intrack(p01):
#             candidates += [s01]

#         print s0, a, candidates
#         i = rnd.choice(len(candidates))
#         return candidates[i]

#     @memoizemethod
#     def R(self, a, s0, s1):
#         if a not in self.env.actions(s0):
#             return 0

#         dvi, dvj = a
#         i0, j0, vi0, vj0 = s0
#         i1, j1, vi1, vj1 = s1

#         p01 = i0 + vi0 + dvi, j0 + vj0 + dvj
#         v01 = vi0 + dvi, vj0 + dvj
#         s01 = p01 + v01

#         return -1 if self.env.intrack(p01) else -5


# track1 = np.array([
#     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
# ]).astype(np.bool)


# if __name__ == '__main__':
#     np.set_printoptions(suppress=True, precision=3, linewidth=np.inf)

#     track = track1
#     maxvel = 4
#     eps = .05

#     env = racetrack_environment(track[::-1, :], maxvel)
#     mod = racetrack_model(env)
#     Q = actionvalues(env)
#     policy = egreedy(env, eps)

#     gamma = 1

#     for s in env.states(begin=True, middle=False, terminal=False):
#         print s, policy[s]

#     op_mc = on_policy_mc(env, mod, policy, Q, gamma)
#     op_mc.run()

#     for s in env.states(begin=True, middle=False, terminal=False):
#         print s, policy[s]


#     plt.matshow(track)
#     for i in xrange(1):
#         episodes = op_mc.sim.run()
#         for s0, _, _, s1 in episodes:
#             i0, j0, _, _ = s0
#             i1, j1, _, _ = s1
#             plt.plot([i0, i1], [j0, j1])

#     plt.show()

#     # print policy

from __future__ import division

import itertools as itt
from collections import defaultdict

import numpy as np
import numpy.random as rnd

from pytk.decorators import memoizemethod

from rl.problems import State, Action, taction, Model, System
from rl.values import Values_Tabular
from rl.policy import Policy_egreedy
from rl.algo.mc import MC


from pytk.util import true_every

# from rl.env import environment, model, ModelException
# from rl.values import actionvalues
# from rl.algo.mc import on_policy_mc
# from rl.policy import egreedy

# import matplotlib.pyplot as plt


# class racetrack_environment(environment):
#     def __init__(self, track, maxvel):
#         self.track = track
#         self.vels = [(vi, vj) for vi in xrange(maxvel+1) for vj in xrange(maxvel+1) if vi != 0 or vj != 0]

#         nrows, ncols = track.shape

#         self.__states_begin = [(0, j, 1, 0) for j in xrange(ncols) if track[0, j]]
#         self.__states_middle = [(i, j, vi, vj) for i in xrange(nrows) for j in xrange(ncols) for vi, vj in self.vels]
#         self.__states_terminal = [(i, ncols-1, vi, vj) for i in xrange(nrows) if track[i, -1] for vi, vj in self.vels]

#         self.__actions = [(dvi, dvj) for dvi in (-1, 0, 1) for dvj in (-1, 0, 1)]

#     @memoizemethod
#     def intrack(self, pos):
#         try:
#             return bool(track[pos])
#         except IndexError:
#             return False

#     @memoizemethod
#     def states(self, begin=True, middle=True, terminal=False):
#         return begin * self.__states_begin \
#                 + middle * self.__states_middle \
#                 + terminal * self.__states_terminal

#     @memoizemethod
#     def actions(self, s):
#         i, j, vi, vj = s
#         return [(dvi, dvj) for dvi, dvj in self.__actions if (vi+dvi, vj+dvj) in self.vels]

# class racetrack_model(model):
#     @memoizemethod
#     def sample_s1(self, a, s0):
#         if a not in self.env.actions(s0):
#             raise ModelException('Action not available')

#         dvi, dvj = a
#         i0, j0, vi0, vj0 = s0

#         # which state does s0 + a amount to?
#         p1 = i0 + vi0 + dvi, j0 + vj0 + dvj
#         v1 =      vi0 + dvi,      vj0 + dvj
#         s1 = p1 + v1

#         if self.env.intrack(p1):
#             return s1

#         candidates = []

#         # move horizontal
#         p01 = i0 + 0, j0 + 1
#         # v01 = vi0 + dvi, vj0 + dvj
#         v01 = vi0, vj0
#         s01 = p01 + v01

#         if self.env.intrack(p01):
#             candidates += [s01]

#         # move vertical
#         p01 = i0 + 1, j0 + 0
#         # v01 = vi0 + dvi, vj0 + dvj
#         v01 = vi0, vj0
#         s01 = p01 + v01

#         if self.env.intrack(p01):
#             candidates += [s01]

#         print s0, a, candidates
#         i = rnd.choice(len(candidates))
#         return candidates[i]

#     @memoizemethod
#     def R(self, a, s0, s1):
#         if a not in self.env.actions(s0):
#             return 0

#         dvi, dvj = a
#         i0, j0, vi0, vj0 = s0
#         i1, j1, vi1, vj1 = s1

#         p01 = i0 + vi0 + dvi, j0 + vj0 + dvj
#         v01 = vi0 + dvi, vj0 + dvj
#         s01 = p01 + v01

#         return -1 if self.env.intrack(p01) else -5


class RacetrackState(State):
    discrete = True

    def __init__(self, pos, vel, terminal=False):
        pos = np.asarray(pos)
        vel = np.asarray(vel)

        self.setkey((str(pos.data), str(vel.data)))

        self.pos = pos
        self.vel = vel
        self.terminal = terminal

    def __str__(self):
        return 'S(pos={}, vel={})'.format(self.pos, self.vel)


class RacetrackAction(Action):
    discrete = True

    def __init__(self, dvel):
        dvel = np.asarray(dvel)

        self.setkey((str(dvel.data),))
        self.dvel = dvel

    def __str__(self):
        return 'A({})'.format(self.dvel)


class RacetrackModel(Model):
    def __init__(self, sys):
        self.sys = sys

    def sample_s0(self):
        y = rnd.choice(self.sys.start_y)
        return RacetrackState([0, y], [1, 0])

    def pr_s1(self, s0, a, s1=None):
        vel = s0.vel + a.dvel
        pos = s0.pos + vel

        if all(pos < self.sys.track.shape):
            if not self.sys.track[pos[0], pos[1]]:
                if self.sys.track[s0.pos[0] + 1, s0.pos[1] + 1]:
                    pos = s0.pos + 1
                elif self.sys.track[s0.pos[0] + 1, s0.pos[1]]:
                    pos = s0.pos + [1, 0]
                else:
                    pos = s0.pos + [0, 1]
                vel = np.array([1, 1])
        elif pos[0] >= self.sys.track.shape[0]:
            # out of bounds top side
            pos[0] = self.sys.track.shape[0] - 1
            vel = np.array([1, 1])
        else:  #  pos[1] >= self.sys.track.shape[1]:
            # out of bounds left side (i.e. vistory)
            pos[1] = self.sys.track.shape[1] - 1

        pr_dict = defaultdict(int, {
            RacetrackState(pos, vel, self.sys.is_terminal(pos)): 1.
        })
        return pr_dict if s1 is None else pr_dict[s1]

    def sample_r(self, s0, a, s1):
        vel = s0.vel + a.dvel
        pos = s0.pos + vel

        return (-1
                if s1.terminal or (all(s0.pos < self.sys.track.shape) and self.sys.track[pos[0], pos[1]])
                else -5)


class RacetrackSystem(System):
    def __init__(self, track, maxvel):
        super(RacetrackSystem, self).__init__(RacetrackModel(self))

        self.track = track
        self.maxvel = maxvel

        self.start_y = np.where(track[0])[0]
        self.end_x = np.where(track[:, -1])[0]
        self.statelist_start = [RacetrackState([0, y], [0, 1]) for y in self.start_y]

        poss = np.column_stack(np.where(track))
        vels = range(maxvel+1)
        vels = [vel for vel in itt.product(vels, vels)][1:]
        self.statelist = [
            RacetrackState(pos, vel, terminal=self.is_terminal(pos))
                for pos in poss for vel in vels
        ]

        dvels = [-1, 0, 1]
        dvels = np.array(list(itt.product(dvels, dvels)))
        self.actionlist = map(RacetrackAction, dvels)

    # @memoizemethod
    # def actions(self, s):
    #     if s.terminal:
    #         return [taction]
    #     return self.actionlist
    #     # TODO
    #     # ...
    #     # return ...

    def is_terminal(self, pos):
        return pos[1] == self.track.shape[1] - 1 and pos[0] in self.end_x


# if __name__ == '__main__':
if False:
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


def make_track(track):
    trackmap = {'#': False , '.': True}
    lines = track.split()
    lines = [map(trackmap.__getitem__, line) for line in lines]
    return np.array(lines)[::-1]


track1 = """
###..............
##...............
##...............
#................
.................
.................
..........#######
.........########
.........########
.........########
.........########
.........########
.........########
.........########
#........########
#........########
#........########
#........########
#........########
#........########
#........########
#........########
##.......########
##.......########
##.......########
##.......########
##.......########
##.......########
##.......########
###......########
###......########
###......########
"""


track2 = """
################................
#############...................
############....................
###########.....................
###########.....................
###########.....................
###########.....................
############....................
#############...................
##############................##
##############.............#####
##############............######
##############..........########
##############.........#########
#############..........#########
############...........#########
###########............#########
##########.............#########
#########..............#########
########...............#########
#######................#########
######.................#########
#####..................#########
####...................#########
###....................#########
##.....................#########
#......................#########
.......................#########
.......................#########
.......................#########
"""


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3, linewidth=np.inf)

    # track = make_track(track1)
    track = make_track(track2)
    maxvel = 4
    eps = .05

    # print track
    # print np.column_stack(np.where(track))

    sys = RacetrackSystem(track, maxvel)
    sys.model.gamma = 1.


    vels = range(maxvel+1)
    print vels
    vels = [vel for vel in itt.product(vels, vels)][1:]
    print vels

    # sys = RacetrackSystem(track[::-1, :], maxvel)
    # model = sys.model

    # Q =
    # policy = Policy_egreedy(Q)

    Q = Values_Tabular.Q()
    policy = Policy_egreedy.Q(Q, .1)

    mc = MC(sys, sys.model, policy, Q)

    verbose = true_every(100)

    nepisodes = 2000
    for i in xrange(nepisodes):
        s0 = sys.model.sample_s0()
        mc.run(s0, verbose.true)

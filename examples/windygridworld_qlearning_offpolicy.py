#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from rl.problems import State, Action, Dynamics, Task, Model
from rl.problems.mdp import MDP
from rl.values import Values_TabularCounted, Values_LinearBayesian
from rl.policy import Policy_UCB, Policy_egreedy
from rl.algo.mc import MCTS
from rl.algo.td import SARSA, SARSA_l, Qlearning

from pytk.util import true_every
import pytk.itt as mytt
import pytk.nptk as nptk

import tkinter as tk
from tkinter import ttk

import threading
import queue
import time


class WindyGridworldState(State):
    discrete = True

    def __init__(self, pos):
        pos = np.asarray(pos)
        self.setkey((str(pos.data),))
        self.pos = pos
        self.tpos = tuple(pos)

    def __str__(self):
        return 'S({})'.format(self.pos)


class WindyGridworldAction(Action):
    discrete = True

    dpos = {
        'up': [-1, 0],
        'down': [1, 0],
        'left': [0, -1],
        'right': [0, 1],
    }

    chars = {
        'up': u'↑',
        'down': u'↓',
        'left': u'←',
        'right': u'→',
    }

    chars = {
        'up': u'⇧',
        'down': u'⇩',
        'left': u'⇦',
        'right': u'⇨',
    }

    def __init__(self, dir_):
        self.setkey((dir_,))

        self.dir_ = dir_
        self.dpos = WindyGridworldAction.dpos[dir_]
        self.char = WindyGridworldAction.chars[dir_]

    def __str__(self):
        return 'A({})'.format(self.dir_)


class WindyGridworldDynamics(Dynamics):
    def __init__(self, gridshape, winds, start):
        super(WindyGridworldDynamics, self).__init__()
        self.gridshape = np.asarray(gridshape)
        self.winds = winds
        self.s0 = WindyGridworldState(start)

    def sample_s0(self):
        return self.s0

    def sample_s1(self, s0, a):
        wind = -self.winds[s0.pos[1]]
        s1pos = s0.pos + a.dpos + [wind, 0]
        s1pos = np.clip(s1pos, [0, 0], self.gridshape-1)
        return WindyGridworldState(s1pos)


class WindyGridworldTask(Task):
    def __init__(self, goal_pos, goal_r):
        super(WindyGridworldTask, self).__init__()
        self.goal_pos = goal_pos
        self.goal_r = goal_r

        self.maxr = max(1, goal_r)

    def sample_r(self, s0, a, s1):
        if self.is_terminal(s1):
            return self.goal_r
        return -1.

    def is_terminal(self, s):
        return np.array_equal(s.pos, self.goal_pos)


class WindyGridworldApp(object):
    def __init__(self, master, gridshape):
        self.gridshape = gridshape
        self.h, self.w = gridshape

        self.frame = ttk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=tk.YES)

        self.canvas = Canvas(self.frame)
        self.canvas.pack(fill=tk.BOTH, expand=YES)

        self.iddict = {}
        for r, c in mytt.grid(gridshape):
            tags = (
                'rect',
                'r{}c{}'.format(r, c),
                'r{}'.format(r),
                'c{}'.format(c),
            )
            color = '#0{}{}'.format(int(10 * r / self.h), int(10 * c / self.w))
            rectid = self.canvas.create_rectangle((0, 0, 0, 0), fill=color, width=2, tags=tags)
            self.iddict[rectid] = r, c

        self.frame.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        h = event.height / self.h
        w = event.width / self.w
        for rectid in self.canvas.find_withtag('rect'):
            r, c = self.iddict[rectid]
            color = '#f{}{}'.format(int(10 * r / self.h), int(10 * c / self.w))
            self.canvas.coords(rectid, c * w, r * h, (c + 1) * w, (r + 1) * h)



class WindyGridworldMDP(MDP):
    """ Dice game MDP. """
    def __init__(self, dynamics, task, gridshape):
        super(WindyGridworldMDP, self).__init__(Model(dynamics, task))

        self.statelist = [WindyGridworldState((i, j)) for i, j in mytt.grid(gridshape)]
        self.actionlist = map(WindyGridworldAction, ('up', 'down', 'left', 'right'))


class WindyGridworldFrame_nvisits(ttk.Frame, object):
    def __init__(self, master, mdp, gridshape, winds, start, goal):
        super(WindyGridworldFrame_nvisits, self).__init__(master)
        self.pack(fill=tk.BOTH, expand=tk.YES)

        self.mdp = mdp
        self.gridshape = gridshape
        self.winds = winds
        self.start = start
        self.goal = goal

        self.nvisits = np.zeros(gridshape, dtype=np.int)
        self.nvisits_mutex = threading.Lock()

        style = ttk.Style()
        style.configure('windy.nframe.TLabel', anchor='center', background='lightblue')
        style.configure('nframe.TLabel', anchor='center', relief='solid')
        style.configure('start.nframe.TLabel', background='green')
        style.configure('goal.nframe.TLabel', background='purple')

        for r in xrange(gridshape[0]+1):
            self.rowconfigure(r, weight=1)
        for c in xrange(gridshape[1]):
            self.columnconfigure(c, weight=1)

        self.svardict = dict()
        self.labeldict = dict()
        for r, c in mytt.grid(gridshape):
            svar = tk.StringVar()
            # svar.set('{}, {}'.format(r, c))
            svar.set('{:4}'.format(0))
            self.svardict[r, c] = svar

            label = ttk.Label(self, textvariable=svar)
            label.grid(row=r, column=c, sticky='nsew')
            self.labeldict[r, c] = label

            if (r, c) == start:
                label.configure(style='start.nframe.TLabel')
            elif (r, c) == goal:
                label.configure(style='goal.nframe.TLabel')
            else:
                label.configure(style='nframe.TLabel')

        for c in xrange(gridshape[1]):
            label = ttk.Label(self, text='{}'.format(winds[c]))
            label.grid(row=gridshape[0], column=c, sticky='nsew')
            label.configure(style='windy.nframe.TLabel')

    def updateloop(self):
        for r, c in mytt.grid(self.gridshape):
            n = self.nvisits[r, c]
            self.svardict[r, c].set('{:4}'.format(n))

        self.after(33, self.updateloop)


class WindyGridworldFrame_policies(ttk.Frame, object):
    def __init__(self, master, mdp, gridshape, winds, start, goal, goals, Qs):
        super(WindyGridworldFrame_policies, self).__init__(master)
        self.pack(fill=tk.BOTH, expand=tk.YES)
        goals = [goal] + goals

        self.mdp = mdp
        self.gridshape = gridshape
        self.winds = winds
        self.start = start
        self.goal = goal
        self.goals = goals
        self.Qs = Qs

        self.task_ind = 0
        ntasks = len(goals)
        self.actiongrid = np.empty((ntasks,) + gridshape, dtype=object)
        # self.actiongrid = np.empty(gridshape, dtype=object)
        self.actiongrid.fill(None)
        self.actiongrid_mutex = threading.Lock()

        style = ttk.Style()
        style.configure('pframe.TLabel', anchor='center', relief='flat')
        style.configure('windy.pframe.TLabel', background='lightblue', relief='solid')
        style.configure('start.pframe.TLabel', background='green', relief='solid')
        style.configure('goals.pframe.TLabel', relief='solid')
        style.configure('goal.goals.pframe.TLabel', background='purple')

        for r in xrange(gridshape[0]+1):
            self.rowconfigure(r, weight=1)
        for c in xrange(gridshape[1]):
            self.columnconfigure(c, weight=1)

        self.svardict = dict()
        self.labeldict = dict()
        for r, c in mytt.grid(gridshape):
            svar = tk.StringVar()
            # svar.set('{}, {}'.format(r, c))
            svar.set('{:4}'.format(0))
            self.svardict[r, c] = svar

            label = ttk.Label(self, textvariable=svar)
            label.grid(row=r, column=c, sticky='nsew')
            self.labeldict[r, c] = label

            if (r, c) == start:
                label.configure(style='start.pframe.TLabel')
            elif (r, c) in goals:
                label.bind('<Button-1>', lambda e, r=r, c=c: self.callback_click(e, r, c))
                if (r, c) == goals[0]:
                    label.configure(style='goal.goals.pframe.TLabel')
                else:
                    label.configure(style='goals.pframe.TLabel')
            else:
                label.configure(style='pframe.TLabel')

        for c in xrange(gridshape[1]):
            label = ttk.Label(self, text='{}'.format(winds[c]))
            label.grid(row=gridshape[0], column=c, sticky='nsew')
            label.configure(style='windy.pframe.TLabel')

    def callback_click(self, event, r, c):
        if (r, c) != self.goals[self.task_ind]:
            event.widget.configure(style='goal.goals.pframe.TLabel')
            label = self.labeldict[self.goal]
            label.configure(style='goals.pframe.TLabel')
            self.task_ind = self.goals.index((r, c))
            self.goal = r, c

    def updateloop(self):
        Q = self.Qs[self.task_ind]
        for s in self.mdp.statelist:
            if s.tpos == self.goal:
                self.svardict[s.tpos].set('')
            else:
                actions = self.mdp.actions(s)
                a = Q.optim_action(s, actions)
                self.svardict[s.tpos].set(a.char)
        # Q = self.Qs[self.task_ind]
        # for r, c in mytt.grid(self.gridshape):
        #     if (r, c) == self.goal:
        #         self.svardict[r, c].set('')
        #     else:
        #         a = self.actiongrid[self.task_ind][r, c]
        #         if a is None:
        #             break
        #         self.svardict[r, c].set(a.char)
        self.after(33, self.updateloop)


if __name__ == '__main__':
    import numpy.random as rnd
    rnd.seed(0)

    gridshape = 7, 10
    winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    start = 3, 0
    goal = 3, 7
    goal_r = 10.

    dynamics = WindyGridworldDynamics(gridshape, winds, start)
    task = WindyGridworldTask(goal, goal_r)
    mdp = WindyGridworldMDP(dynamics, task, gridshape)

    Q = Values_TabularCounted.Q()
    # policy = Policy_UCB.Q(Q.value, Q.confidence, beta=task.maxr)
    policy = Policy_egreedy.Q(Q, .01)

    goals = [
        (0, 0),
        (2, 6),
        (2, 8),
        (4, 6),
        (4, 8),
    ]
    Q_tasks = [(Values_TabularCounted.Q(), WindyGridworldTask(goal_, goal_r)) for goal_ in goals]
    algo = Qlearning(mdp, policy, Q, Q_tasks)

    Qs = [Q] + [Q_ for Q_, _ in Q_tasks]

    frame_nvisits = None
    root = tk.Tk()
    frame_nvisits = WindyGridworldFrame_nvisits(root, mdp, gridshape, winds, start, goal)

    frame_policies = None
    topl = tk.Toplevel()
    frame_policies = WindyGridworldFrame_policies(topl, mdp, gridshape, winds, start, goal, goals, Qs)

    root.update()

    def target():

        def callback_step(*args, **kwargs):
            if frame_nvisits is not None:
                s1 = kwargs['s1']
                pos = tuple(s1.pos)
                # with frame_nvisits.nvisits_mutex:
                frame_nvisits.nvisits[pos] += 1

            # if frame_policies is not None:
            #     Qs = kwargs['Qs']
            #     # with frame_policies.actiongrid_mutex:
            #     Q = Qs[frame_policies.task_ind]
            #     for s in mdp.statelist:
            #         actions = mdp.actions(s)
            #         frame_policies.actiongrid[s.tpos] = Q.optim_action(s, actions)

            # if frame_policies is not None:
            #     Qs = kwargs['Qs']
            #     for i, Q in enumerate(Qs):
            #         for s in mdp.statelist:
            #             actions = mdp.actions(s)
            #             frame_policies.actiongrid[i][s.tpos] = Q.optim_action(s, actions)

            # HACK! GUI does not respond correctly if I don't insert a short break
            # NOTE OH it's the GIL, of course..
            # time.sleep(.5e-3)
            time.sleep(1e-4)
            # time.sleep(5e-4)

        nepisodes = 1000

        for i in xrange(nepisodes):
            print 'running episode: {:5} / {:5}'.format(i, nepisodes)
            s0 = dynamics.sample_s0()
            algo.run(s0, callback_step=callback_step)

    thread = threading.Thread(target=target)
    thread.setDaemon(True)
    thread.start()

    if frame_nvisits is not None:
        frame_nvisits.updateloop()
    if frame_policies is not None:
        frame_policies.updateloop()

    root.mainloop()

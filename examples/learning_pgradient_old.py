#!/usr/bin/env python

import logging.config
from logconfig import LOGGING

import rl.pomdp as pomdp
import rl.pomdp.policies as policies
import rl.pomdp.algos as algos
import rl.optim as optim
import rl.graph as graph
import pyqtgraph as pg

import numpy as np

import multiprocessing as mp
from tqdm import tqdm

import argparse


# TODO get rid of this monstruorisy?
class StepSizeAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        nmin, nmax = 1, 2
        if not nmin <= len(values) <= nmax:
            raise argparse.ArgumentTypeError(
                f'argument "{self.dest}" requires between {nmin} and '
                f'{nmax} arguments'
            )

        try:
            s0, decay = values
        except IndexError:
            s0, decay = values[0], None

        stepsize = optim.StepSize(s0, decay=decay)
        setattr(args, self.dest, stepsize)


class ObjectiveAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        weights, objectives = [], []

        i = 0
        while i < len(values):
            try:
                w = float(values[i])
            except ValueError:
                weights.append(1.)
                objectives.append(values[i])
                i += 1
            else:
                weights.append(w)
                objectives.append(values[i + 1])
                i += 2

        setattr(args, 'weights', np.array(weights, dtype=np.float))
        setattr(args, 'objectives', objectives)


def target_pbar(nruns, nepisodes, nprocesses, q):
    name = mp.current_process().name
    pbar_runs = tqdm(desc=name, total=nruns, position=0)
    desc = 'Worker-{}'.format
    pbar_episodes = [tqdm(desc=desc(wid), total=nepisodes, position=wid + 1)
                     for wid in range(nprocesses)]
    for type_, wid in iter(q.get, None):
        if type_ == 'episode':
            pbar_episodes[wid].update()
        elif type_ == 'run':
            pbar_episodes[wid].close()
            pbar_episodes[wid] = tqdm(desc=desc(wid), total=nepisodes,
                                      position=wid + 1)
            pbar_runs.update()
            pbar_runs.refresh()


if __name__ == '__main__':
    # logging configuration
    logging.config.dictConfig(LOGGING)

    parser = argparse.ArgumentParser(description='Policy Gradient')

    parser.add_argument('--pbar', action='store_true', help='output file name')
    parser.add_argument('--graph', action='store_true', help='show graphics')
    parser.add_argument('--line', nargs=2, action='append', dest='lines',
                        default=[], help='plot lines')
    parser.add_argument(
        '--optimal', type=str, default=None, help='optimal returns')
    parser.add_argument(
        '--out', type=str, default=None, help='output file name')
    # parser.add_argument(
    #     '--ostack', action='store_true', help='stack results')
    # parser.add_argument(
    #     '--owrite', action='store_true', help='stack results')

    parser.add_argument(
        '--processes', type=int, default=mp.cpu_count() - 1,
        help='number of processes')
    parser.add_argument(
        '--runs', type=int, default=10, help='number of training runs')
    parser.add_argument(
        '--episodes', type=int, default=1000,
        help='number of training episodes')
    parser.add_argument(
        '--steps', type=int, default=100,
        help='number of steps in an episode')

    parser.add_argument(
        '--stepsize', type=float, default=None, nargs='+',
        help='step size', action=StepSizeAction)
    parser.add_argument(
        '--clip', type=float, default=None, help='clipping limit')

    # parser.add_argument('--nu', dest='obj',
    #         action='store_const', const='longterm_average',
    #         default='longterm_average')
    # parser.add_argument('--J', dest='obj',
    #         action='store_const', const='discounted_sum')

    parser.add_argument('pomdp', type=str, help='POMDP name')
    parser.add_argument('policy', type=str, help='Policy arguments')
    parser.add_argument(
        'objectives', type=str, nargs='+', action=ObjectiveAction,
        help='Objectives arguments')

    args = parser.parse_args()
    print(f'Argument Namespace: {args}')

    nprocesses = args.processes
    nruns = args.runs
    nepisodes = args.episodes
    nsteps = args.steps
    # obj = args.obj

    clip = args.clip
    clip2 = clip ** 2 if clip is not None else None
    stepsize = args.stepsize

    if stepsize is None:
        stepsize = optim.StepSize(1.)

    # TODO check that algo-policy combination works
    env = pomdp.Environment.from_fname(args.pomdp)

    # TODO how to handle objective?  I guess it's actually handled by the
    # actual objective...

    # TODO objective..... how to? I guess it depends on each "objective"
    # objective = getattr(pomdp.objectives, obj)(env)
    objectives = [algos.factory(obj) for obj in args.objectives]
    nobjectives = len(objectives)
    policy = policies.factory(env, args.policy)

    print(env)
    print(policy)
    print('Objectives:')
    for w, obj, objf in zip(args.weights, args.objectives, objectives):
        print(' *', w, obj, '->', objf)
    print()

    plot_objectives = None
    plot_policy = None
    if args.graph:

        lines = []
        for color, value in args.lines:
            try:
                data = np.load(value)
            except FileNotFoundError:
                lines += [
                    dict(pos=float(value), angle=0, pen=dict(color=color))
                ]
            else:
                datap = np.percentile(data, [0, 25, 50, 75, 100])
                lines += [
                    dict(pos=datap[0], angle=0,
                         pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
                    dict(pos=datap[1], angle=0,
                         pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
                    dict(pos=datap[2], angle=0, pen=dict(color=color)),
                    dict(pos=datap[3], angle=0,
                         pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
                    dict(pos=datap[4], angle=0,
                         pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
                ]

        plot_objectives = graph.Plotter(
            (nobjectives, nepisodes),
            window=dict(text='Objectives', size='16pt', bold=True),
            labels=dict(left='Value', bottom='Episode'),
            lines=lines
        )

        plot_policy = policy.new_plot(nepisodes)

    lock = mp.Lock()
    rvalue = mp.RawValue('i', 0)
    wvalue = mp.RawValue('i', 0)

    def initializer(*args):
        global wid
        with lock:
            wid = wvalue.value
            wvalue.value += 1

        np.random.seed()

    def run(ri, q=None):
        with lock:
            # truly sequential index
            ri = rvalue.value
            rvalue.value += 1

        returns_run = np.empty(nepisodes)
        idx_returns = 0

        params = policy.new_params()
        nparams = params.size
        # nobjectives = len(objectives)

        # TODO I want to store the things independnetly
        shape = nobjectives, nparams
        dparams = np.empty(shape, dtype=object)
        gnorms2 = np.empty(shape, dtype=object)

        stepsize.reset()
        for e in range(nepisodes):
            # NOTE ovalues should be a brand new array, for the queue
            ovalues = np.zeros(nobjectives)
            dparams.fill(0.)

            for oi, (w, objective) in enumerate(zip(args.weights, objectives)):
                # TODO parallelize this!!!!!! not the other one!
                # TODO how to give nsteps?
                # TODO here run main env loop!!!!
                obj, dobj = objective(params, policy, env)
                ovalues[oi] = w * obj
                dparams[oi] = w * dobj

            # for oi in, dp2 in enumerate(np.square(dparams)):
            # dparams2 = np.square(dparams)
            # for i in np.ndindex(dparams.shape):
            #     gnorms2[i] = dparams2[i].sum()
            for idx, dp in np.ndenumerate(np.square(dparams)):
                gnorms2[idx] = dp.sum()
            gnorm2 = gnorms2.sum()

            # TODO now I can do stuff with all this stuff!
            if clip is not None and gnorm2 > clip2:
                gnorm = np.sqrt(gnorm2)
                dparams *= clip / gnorm

            params += stepsize() * dparams.sum(axis=0)

            stepsize.step()

            returns_run[idx_returns] = ovalues[0]
            # print(ovalues, gnorms2.ravel())
            # print(ovalues)

            idx_returns += 1

            if ri == 0:
                # if plot_objectives is not None:
                #     plot_objectives.update(ovalues[0])
                if plot_objectives is not None:
                    plot_objectives.update(ovalues)
                if plot_policy is not None:
                    plot_policy.update(params)

            try:
                q_pbar.put(('episode', wid))
            except AttributeError:
                pass

        try:
            q_pbar.put(('run', wid))
        except AttributeError:
            pass

        if plot_policy is not None:
            plot_policy.close()

        return returns_run

    if args.pbar:
        q_pbar = mp.Queue()
        p_pbar = mp.Process(target=target_pbar,
                            args=(nruns, nepisodes, nprocesses, q_pbar))
        p_pbar.daemon = True
        p_pbar.start()
    else:
        p_pbar, q_pbar = None, None

    if nprocesses > 1:  # parallel
        with mp.Pool(processes=nprocesses, initializer=initializer) as pool:
            rets = pool.map(run, range(nruns))
    else:  # sequential
        initializer()
        rets = [run(ri) for ri in range(nruns)]
    rets = np.array(rets)

    if args.out:
        np.save(args.out, rets)

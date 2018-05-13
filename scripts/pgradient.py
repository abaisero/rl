#!/usr/bin/env python

import sys

# import logging.config
# from logconfig import LOGGING

import rl.pomdp as pomdp
import rl.pomdp.policies as policies
import rl.pomdp.algos as algos
# import rl.optim as optim
import rl.graph as graph
import pyqtgraph as pg

import numpy as np

import multiprocessing as mp
from tqdm import tqdm

import argparse


# # TODO get rid of this monstruorisy?
# class StepSizeAction(argparse.Action):
#     def __call__(self, parser, args, values, option_string=None):
#         nmin, nmax = 1, 2
#         if not nmin <= len(values) <= nmax:
#             raise argparse.ArgumentTypeError(
#                 f'argument "{self.dest}" requires between {nmin} and '
#                 f'{nmax} arguments'
#             )

#         try:
#             s0, decay = values
#         except ValueError:
#             s0, decay = values[0], None

#         stepsize = optim.StepSize(s0, decay=decay)
#         setattr(args, self.dest, stepsize)


class ObjectiveAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        weights, objectives = [], []
        for value in values:
            vsplit = value.split()
            try:
                w = float(vsplit[0])
            except ValueError:
                w = 1.
            else:
                vsplit.pop(0)

            objective = ' '.join(vsplit)

            weights.append(w)
            objectives.append(objective)

        weights = np.array(weights)
        setattr(args, 'weights', weights)
        setattr(args, self.dest, objectives)


def target_pbar(nruns, nepisodes, nprocesses, q):
    def new_pbar(id_worker):
        return tqdm(desc=desc(id_worker), total=nepisodes,
                    position=id_worker + 1)
    name = mp.current_process().name
    pbar_runs = tqdm(desc=name, total=nruns, position=0)
    desc = 'Worker-{}'.format
    pbar_episodes = [new_pbar(id_worker) for id_worker in range(nprocesses)]
    for type_, id_worker in iter(q.get, None):
        if type_ == 'episode':
            pbar_episodes[id_worker].update()
        elif type_ == 'run':
            pbar_episodes[id_worker].close()
            pbar_episodes[id_worker] = new_pbar(id_worker)
            pbar_runs.update()
            pbar_runs.refresh()
        elif type_ == 'end':
            pbar_runs.close()


if __name__ == '__main__':
    # logging configuration
    # logging.config.dictConfig(LOGGING)

    parser = argparse.ArgumentParser(description='Policy Gradient')

    parser.add_argument('--pbar', action='store_true', help='progress bars')
    parser.add_argument('--graph', action='store_true', help='graphics')
    parser.add_argument('--line', metavar=('C', 'F'), nargs=2, action='append',
                        dest='lines', default=[], help='graph reference lines')

    parser.add_argument('--out', metavar='F', type=str, default=None,
                        help='output file name')

    parser.add_argument('--processes', metavar='P', type=int,
                        default=mp.cpu_count() - 1, help='number of processes')

    parser.add_argument('--runs', metavar='R', type=int, default=10,
                        help='number of learning runs')
    parser.add_argument('--episodes', metavar='E', type=int, default=1000,
                        help='number of episodes in run')
    parser.add_argument('--steps', metavar='S', type=int, default=100,
                        help='number of steps in episode')

    parser.add_argument('--stepsize', metavar='SS', type=float, default=1.,
                        help='stepsize')
    parser.add_argument('--clip', metavar='C', type=float, default=None,
                        help='clip limit')

    parser.add_argument('env', type=str, help='environment')
    parser.add_argument('policy', type=str, help='policy')
    parser.add_argument('objs', type=str, nargs='+', action=ObjectiveAction,
                        help='objectives')

    config = parser.parse_args()
    print(f'Argument Namespace: {config}')

    nprocesses = config.processes
    nruns = config.runs
    nepisodes = config.episodes
    nsteps = config.steps

    stepsize = config.stepsize
    clip = config.clip
    clip2 = clip ** 2 if clip is not None else None

    env = pomdp.Environment.from_fname(config.env)
    policy = policies.factory(env, config.policy)
    objectives = [algos.factory(env, policy, obj) for obj in config.objs]
    nobjectives = len(objectives)

    print(f'Environment: {env}')
    print(f'Policy: {policy}')
    print('Objectives:')
    for w, obj, objf in zip(config.weights, config.objs, objectives):
        print(f' - {w} {obj} -> {objf}')
    print()

    plot_objectives = None
    plot_policy = None
    if config.graph:

        lines = []
        for color, value in config.lines:
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

    #
    lock = mp.Lock()
    rvalue_job = mp.RawValue('i', 0)
    rvalue_worker = mp.RawValue('i', 0)

    def initializer():
        global id_worker
        with lock:
            id_worker = rvalue_worker.value
            rvalue_worker.value += 1

        np.random.seed()

    def run(_, q=None):
        with lock:
            # truly sequential index
            idx_job = rvalue_job.value
            rvalue_job.value += 1

        run_objs = np.empty((nobjectives, nepisodes))
        run_gnorms = np.empty((nobjectives, nepisodes))

        params = policy.new_params()
        nparams = params.size

        objs = np.empty(nobjectives)
        gnorms = np.empty(nobjectives)
        grads = np.empty((nobjectives, nparams), dtype=object)
        gnorms2 = np.empty((nobjectives, nparams))

        for idx_episode in range(nepisodes):
            econtext = env.new_context()
            pcontext = policy.new_context(params)
            acontexts = [objective.new_context()
                         if objective.type_ == 'episodic'
                         else None
                         for objective in objectives]
            while econtext.t < nsteps:
                a = policy.sample_a(params, pcontext)
                feedback, econtext1 = env.step(econtext, a)
                pcontext1 = policy.step(params, pcontext, feedback)

                for objective, acontext in zip(objectives, acontexts):
                    if acontext is not None:
                        objective.step(params, acontext, econtext, pcontext,
                                       a, feedback, pcontext1, inline=True)

                econtext = econtext1
                pcontext = pcontext1

            j = 0
            for i, objective in enumerate(objectives):
                if objective.type_ == 'episodic':
                    objs[i] = acontexts[j].obj
                    grads[i] = acontexts[j].grad
                    j += 1
                elif objective.type_ == 'analytic':
                    objs[i], grads[i] = objective(params)
                else:
                    assert False, ('objective.type_ not in '
                                   '[\'episodic\', \'analytic\'].')

            for idx, grad in np.ndenumerate(np.square(grads)):
                gnorms2[idx] = grad.sum()
            gnorm2 = gnorms2.sum()
            gnorms = np.sqrt(gnorms2.sum(axis=1))

            if clip is not None and gnorm2 > clip2:
                grads *= clip / np.sqrt(gnorm2)

            losses = config.weights * objs
            dloss = np.dot(config.weights, grads)
            params += stepsize * dloss

            run_objs[:, idx_episode] = objs
            run_gnorms[:, idx_episode] = gnorms

            if idx_job == 0:
                # TODO graph gnorms

                # if plot_objectives is not None:
                #     plot_objectives.update(losses[0])
                if plot_objectives is not None:
                    plot_objectives.update(losses)
                if plot_policy is not None:
                    plot_policy.update(params)

            try:
                q_pbar.put(('episode', id_worker))
            except AttributeError:
                pass

        try:
            q_pbar.put(('run', id_worker))
        except AttributeError:
            pass

        if plot_policy is not None:
            plot_policy.close()

        return run_objs, run_gnorms

    if config.pbar:
        q_pbar = mp.Queue()
        p_pbar = mp.Process(target=target_pbar,
                            args=(nruns, nepisodes, nprocesses, q_pbar))
        p_pbar.daemon = True
        p_pbar.start()
    else:
        p_pbar, q_pbar = None, None

    # TODO return multiple things
    if nprocesses > 1:  # parallel
        with mp.Pool(processes=nprocesses, initializer=initializer) as pool:
            jobs = pool.map(run, range(nruns))
    else:  # sequential
        initializer()
        jobs = [run(idx_job) for idx_job in range(nruns)]
    jobs = np.array(jobs)

    try:
        q_pbar.put(('end', None))
    except AttributeError:
        pass

    objs = jobs[:, 0, ...]
    gnorms = jobs[:, 1, ...]

    if config.out:
        np.savez(config.out, objs=objs, gnorms=gnorms, weights=config.weights)

        args = '\n'.join(sys.argv)
        with open(f'{config.out}.txt', 'w') as f:
            print('# sys.argv', file=f)
            print(sys.argv, file=f)
            print(file=f)
            print('# args', file=f)
            print(args, file=f)
            print(file=f)
            print('# config', file=f)
            print(config, file=f)

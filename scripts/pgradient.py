#!/usr/bin/env python

import sys
import logging
import logging.config
import argparse

import rl.data as data
import rl.pomdp as pomdp
import rl.pomdp.policies as policies
import rl.pomdp.algos as algos
import rl.optim as optim
import rl.graph as graph
import pyqtgraph as pg

import numpy as np

import multiprocessing as mp
from tqdm import tqdm


# # TODO get rid of this monstruorisy?  Probably create set of optimizers stuff
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


# TODO move elsewhere?
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
    parser = argparse.ArgumentParser(description='Policy Gradient')

    parser.add_argument('--pbar', action='store_true', help='progress bars')
    parser.add_argument('--graph_objectives', action='store_true',
                        help='graphics')
    parser.add_argument('--graph_policy', action='store_true', help='graphics')
    parser.add_argument('--gref', metavar=('C', 'F'), nargs=2, action='append',
                        dest='grefs', default=[], help='graph reference lines')

    parser.add_argument('--out', metavar='F', type=str, default=None,
                        help='output file name')

    parser.add_argument('--processes', metavar='P', type=int,
                        default=mp.cpu_count() - 1, help='number of processes')
    parser.add_argument('--samples', metavar='S', type=int,
                        default=1, help='number of MC samples')

    parser.add_argument('--runs', metavar='R', type=int, default=10,
                        help='number of learning runs')
    parser.add_argument('--episodes', metavar='E', type=int, default=1000,
                        help='number of episodes in run')
    parser.add_argument('--steps', metavar='S', type=int, default=100,
                        help='number of steps in episode')

    parser.add_argument('--adam', help='Adam', action='store_true')
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

    try:
        level = getattr(logging, config.log.upper())
    except AttributeError:
        level = None

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,

        'formatters': {
            'simple': {
                'format': '{levelname} {name} {message}',
                'style': '{',
            },
        },

        'handlers': {
            'file': {
                'level': level,
                'class': 'logging.FileHandler',
                'filename': 'pgradient.log',
                'mode': 'w',
                'formatter': 'simple',
            },
        },

        'loggers': {
            'rl': {
                'handlers': ['file'],
                'level': 'DEBUG',
                'propagate': True,
            },
        },
    })

    logger = logging.getLogger('rl')
    logger.info(f'Running evaluate.py')
    logger.info(f' - args:  {sys.argv[1:]}')
    logger.info(f' - config:  {config}')

    nprocesses = config.processes
    nsamples = config.samples
    nruns = config.runs
    nepisodes = config.episodes
    nsteps = config.steps

    if config.adam:
        optimizer = optim.Adam()
    else:
        optimizer = optim.GDescent(config.stepsize, config.clip)

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
    plot_gnorms = None
    plot_policy = None
    if config.graph_objectives:

        # TODO represent the reference lines somewhere else...
        lines = []
        for color, refname in config.grefs:
            data_gref = np.load(data.resource_path(refname, 'grefs'))
            pdata_gref = np.percentile(data_gref, [0, 25, 50, 75, 100])
            lines += [
                dict(pos=pdata_gref[0], angle=0,
                     pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
                dict(pos=pdata_gref[1], angle=0,
                     pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
                dict(pos=pdata_gref[2], angle=0, pen=dict(color=color)),
                dict(pos=pdata_gref[3], angle=0,
                     pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
                dict(pos=pdata_gref[4], angle=0,
                     pen=dict(color=color, style=pg.QtCore.Qt.DotLine)),
            ]

        plot_objectives = graph.Plotter(
            (nobjectives, nepisodes),
            window=dict(text='Objectives', size='16pt', bold=True),
            labels=dict(left='Value', bottom='Episode'),
            lines=lines
        )

        plot_gnorms = graph.Plotter(
            (nobjectives, nepisodes),
            window=dict(text='Grad Norms', size='16pt', bold=True),
            labels=dict(left='GNorm', bottom='Episode'),
        )

    if config.graph_policy:
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
            objs.fill(0)
            grads.fill(0)

            for _ in range(nsamples):
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
                            objective.step(params, acontext, econtext,
                                           pcontext, a, feedback, pcontext1,
                                           inline=True)

                    econtext = econtext1
                    pcontext = pcontext1

                for i, (objective, acontext) \
                        in enumerate(zip(objectives, acontexts)):
                    if objective.type_ == 'episodic':
                        obj, grad = acontext.obj, acontext.grad
                    elif objective.type_ == 'analytic':
                        obj, grad = objective(params)
                    else:
                        assert False, ('objective.type_ not in '
                                       '[\'episodic\', \'analytic\'].')
                    objs[i] += obj
                    grads[i] += grad

            objs /= nsamples
            grads /= nsamples

            # TODO definitely better way
            for idx, grad in np.ndenumerate(np.square(grads)):
                gnorms2[idx] = grad.sum()
            # gnorm2 = gnorms2.sum()
            gnorms = np.sqrt(gnorms2.sum(axis=1))

            objs_weighted = config.weights * objs
            gnorms_weighted = config.weights * gnorms
            grad_weighted = np.dot(config.weights, grads)

            params += optimizer(grad_weighted)
            policy.process_params(params, inline=True)

            run_objs[:, idx_episode] = objs
            run_gnorms[:, idx_episode] = gnorms

            if idx_job == 0:
                if plot_objectives is not None:
                    plot_objectives.update(objs_weighted)
                if plot_gnorms is not None:
                    plot_gnorms.update(gnorms_weighted)
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
        # save results
        np.savez(config.out, objs=objs, gnorms=gnorms, weights=config.weights)

        # save cmdline
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

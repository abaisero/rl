import rl.pomdp as pomdp
import rl.pomdp.envs as envs
import rl.pomdp.policies as policies
import rl.pomdp.psearch as psearch
import rl.pomdp.agents as agents
import rl.optim as optim
import rl.graph as graph
import rl.misc as misc

from pytk.callbacks import Callback

import numpy as np
import numpy.random as rnd

import logging.config
from logconfig import LOGGING

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import multiprocessing as mp
import time


if __name__ == '__main__':
    # np.seterr(all='raise')

    # logging configuration
    logging.config.dictConfig(LOGGING)

    nruns, nepisodes, horizon = 100, 500, 100
    shape = nruns, nepisodes

    envname = 'Tiger'
    # envname = 'loadunload'
    # envname = 'heaven-hell'
    # envname = 'Hallway'
    # envname = 'Hallway2'
    # envname = 'TagAvoid'  # funny;  probabilities don't sum up to 1
    with envs.dotpomdp(envname) as f:
        env = envs.parse(f)

    # env = envs.Tiger(.01)
    # env.gamma = .95

    N = 10
    # beta = .95
    beta = .5
    step_size = optim.StepSize(.01)
    # step_size = optim.StepSize(1)
    # step_size = optim.Geometric(10, .999)
    eps = 1e-10
    processes = mp.cpu_count()
    processes = mp.cpu_count() - 2

    # TODO maybe FSC needs different step sizes for different strategies?

    # Random
    # policy = policies.Random(env)
    # agent = agents.Agent('Random', env, policy)

    # TODO blind should also use the policy gradient interface....
    # TODO blind should just be a type of policy...... then just use GPOMDP

    # Blind (tries to learn best action distribution)
    # policy = policies.Blind(env)
    # agent = agents.Blind('Blind', env, policy)

    # GPOMDP
    # policy = policies.Reactive(env)
    # pg = psearch.GPOMDP(policy, beta)
    # name = f'GPOMDP ($\\beta$={beta})'
    # agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # CONJGPOMDP + GPOMDP
    # policy = policies.Reactive(env)
    # pg = psearch.GPOMDP(policy, beta)
    # ps = psearch.CONJPOMDP(policy, pg, step_size=step_size, eps=eps)
    # name = f'CONJPOMDP-GPOMDP ($\\beta$={beta}, $\\epsilon$={eps})'
    # agent = agents.PolicySearch(name, env, policy, ps)

    # Istate-GPOMDP (params N and beta)
    policy = policies.FSC(env, N)
    pg = psearch.IsGPOMDP(policy, beta)
    name = f'IsGPOMDP (N={N}, $\\beta$={beta})'
    agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # CONJGPOMDP + Istate-GPOMDP
    # policy = policies.FSC(env, N)
    # pgrad = psearch.IsGPOMDP(policy, beta)
    # ps = psearch.CONJPOMDP(policy, pgrad, step_size=step_size, eps=eps)
    # name = f'CONJPOMDP-IsGPOMDP (N={N}, $\\beta$={beta}, $\\epsilon$={eps})'
    # agent = agents.PolicySearch(name, env, policy, ps)

    def pdict_item(percentile, **kwargs):
        return percentile, dict(name=f'{percentile/100:.2f}', **kwargs)

    from pyqtgraph.Qt import QtCore

    pdict = dict([
        pdict_item(100, pen=dict(color='g', style=QtCore.Qt.DotLine)),
        pdict_item( 75, pen=dict(color='g')),
        pdict_item( 50, pen=dict(color='w')),
        pdict_item( 25, pen=dict(color='r')),
        pdict_item(  0, pen=dict(color='r', style=QtCore.Qt.DotLine)),
    ])

    q_returns, _ = graph.pplot(shape, pdict,
        window=dict(text='Returns', size='16pt', bold=True),
        labels=dict(left='G_t', bottom='Episode'),
    )
    q_gnorms, _ = graph.pplot(shape, pdict,
        window=dict(text='Gradient Norms', size='16pt', bold=True),
        labels=dict(left='|w|', bottom='Episode'),
        # ranges=dict(y=[0, None]),
    )

    # ashape = nepisodes, policy.nnodes, env.nactions
    # q_amodel, _ = graph.distplot(ashape,
    #     window=dict(text='Action Strategies', size='16pt', bold=True),
    #     ylabels=env.afactory.values,
    #     xlabels=policy.nfactory.values
    # )

    q_fsc, _ = graph.fscplot(policy, nepisodes)

    # oshape = nepisodes, policy.nnodes, nnodes
    # q_omodel, _ = graph.distplot(oshape,
    #     window=dict(text='Observation Strategies', size='16pt', bold=True),
    #     ylabels=env.afactory.values,
    #     xlabels=policy.nfactory.values
    # )

    H = misc.Horizon(horizon)
    sys = pomdp.System(env, env.model, H)

    v = mp.RawValue('i', 0)
    l = mp.Lock()
    def run(ri):
        with l:
            # this index is a better one!! linear throughout processes
            i = v.value
            v.value += 1

        seed = int(time.time() * 1000 + i * 61001) % 2 ** 32
        print(f'Starting run {i+1} / {nruns};  Running {nepisodes} episodes... (with seed {seed})')

        rnd.seed(seed)  # ensure different randomization

        idx_results = i * nepisodes
        idx_gnorms = i * nepisodes
        # idx_amodel = 0
        idx_fsc = 0

        # NOTE if this is outside I have a bug
        @Callback
        def episode_return(sys, episode):
            G = 0.
            for context, a, feedback in episode:
                r = feedback.r
                G = r + env.gamma * G
            return G

        feedbacks = ()
        feedbacks_episode = episode_return,

        if i == 0:
            def episode_dplot(sys, episode):
                global policy
                # nonlocal idx_amodel
                nonlocal idx_fsc

                adist = policy.amodel.probs()
                # TODO bug already happening...
                adist /= adist.sum(axis=-1, keepdims=True)
                # q_amodel.put((idx_amodel, dist))

                odist = policy.omodel.probs()
                odist /= odist.sum(axis=-1, keepdims=True)

                q_fsc.put((idx_fsc, adist, odist))
                idx_fsc += 1

                if idx_fsc == nepisodes:
                    q_fsc.put(None)

            feedbacks_episode += episode_dplot,

        # TODO I can still print the different between parameters!!!!
        # YES this is correct..
        def episode_gnorm(gradient):
            nonlocal idx_gnorms
            if gradient.dtype == object:
                gnorm = np.sqrt(sum(_.sum() for _ in gradient ** 2))
            else:
                gnorm = np.sqrt(np.sum(gradient ** 2))
            q_gnorms.put((idx_gnorms, gnorm))
            idx_gnorms += 1

        # TODO better way to handle callbacks...
        agent.callbacks_episode = [episode_gnorm]
        # TODO... how to do callbacks for different types of agents?

        @episode_return.callback
        def plot_return(G):
            nonlocal idx_results
            q_returns.put((idx_results, G))
            idx_results += 1

        #  reset agent before each new learning run
        agent.reset()
        sys.run(
            agent,
            nepisodes=nepisodes,
            feedbacks=feedbacks,
            feedbacks_episode=feedbacks_episode,
        )


    if processes > 1: # NOTE parallelized
        with mp.Pool(processes=processes) as pool:
            result = pool.map(run, range(nruns))
    else: # NOTE serialized
        for ri in range(nruns):
            run(ri)


    #  keeps figures alive
    import IPython
    IPython.embed()

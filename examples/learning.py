import rl.pomdp as pomdp
import rl.pomdp.envs as envs
import rl.pomdp.policies as policies
import rl.pomdp.psearch as psearch
import rl.pomdp.agents as agents
import rl.optim as optim
import rl.graph as graph
import rl.parsers.dotpomdp as dotpomdp
import rl.parsers.dotfsc as dotfsc
import rl.misc as misc

from pytk.callbacks import Callback

import numpy as np
import numpy.random as rnd

import logging.config
from logconfig import LOGGING

from pyqtgraph.Qt import QtCore

import multiprocessing as mp
import time


if __name__ == '__main__':
    # np.seterr(all='raise')

    # logging configuration
    logging.config.dictConfig(LOGGING)

    nruns, nepisodes, horizon = 10, 5000, 100
    shape = nruns, nepisodes

    # NOTE
    # envname, N, beta, step_size = 'loadunload', 2, .95, optim.StepSize(1)

    envname = 'Tiger.smooth'
    # envname = 'Tiger'
    # envname = 'loadunload'
    # envname = 'heaven-hell'
    # envname = 'Hallway'
    # envname = 'Hallway2'
    # envname = 'TagAvoid'  # funny;  probabilities don't sum up to 1
    # with dotpomdp.open(envname) as f:
    #     env = envs.parse(f)

    env = dotpomdp.env(envname)

    # env = envs.Tiger(.01)
    # env.gamma = .95

    # TODO make script file!
    # TODO make sparse FSC
    # TODO make structured FSC!!!
    # TODO save running results and combine in a table / plot...

    N = 10
    K = 3
    beta = .95
    # step_size = optim.StepSize(1)
    step_size = optim.StepSize(.1)
    # step_size = optim.StepSize(.01)
    # step_size = optim.Geometric(10, .999)
    eps = 1e-10
    # processes = 1
    # processes = mp.cpu_count()
    processes = mp.cpu_count() - 1

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
    # policy = policies.FSC(env, N)
    # pg = psearch.IsGPOMDP(policy, beta)
    # name = f'IsGPOMDP (N={N}, $\\beta$={beta})'
    # agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # Sparse Istate-GPOMDP (params N, K and beta)
    # policy = policies.SparseFSC(env, N, K)
    # pg = psearch.IsGPOMDP(policy, beta)
    # name = f'IsGPOMDP (N={N}, $\\beta$={beta})'
    # agent = agents.PolicyGradient(name, env, policy, pg, step_size=step_size)

    # Structured Istate-GPOMDP (params N and beta)

    fscname = 'tiger.lin.1'  # early success
    fscname = 'tiger.lin.2'  # late success
    fscname = 'tiger.lin.3'  #
    policy = dotfsc.fsc(fscname, env)
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

    pdict = dict([
        pdict_item(  0, pen=dict(color='r', style=QtCore.Qt.DotLine)),
        pdict_item(100, pen=dict(color='g', style=QtCore.Qt.DotLine)),
        pdict_item( 25, pen=dict(color='r')),
        pdict_item( 75, pen=dict(color='g')),
        pdict_item( 50),
    ])

    q_returns, _ = graph.pplot(shape, pdict,
        window=dict(text='Returns', size='16pt', bold=True),
        labels=dict(left='G_t', bottom='Episode'),
    )
    q_gnorms, _ = graph.pplot(shape, pdict,
        window=dict(text='Gradient Norms', size='16pt', bold=True),
        labels=dict(left='|w|', bottom='Episode'),
    )

    try:
        pplot = policy.plot
    except AttributeError:
        policy_plot = False
    else:
        pplot(nepisodes)
        policy_plot = True

    H = misc.Horizon(horizon)
    sys = pomdp.System(env, env.model, H)

    v = mp.RawValue('i', 0)
    l = mp.Lock()
    def run(ri):
        with l:
            # true sequential index
            i = v.value
            v.value += 1

        seed = int(time.time() * 1000 + i * 61001) % 2 ** 32
        print(f'Starting run {i+1} / {nruns};  Running {nepisodes} episodes... (with seed {seed})')

        rnd.seed(seed)  # ensure different randomization

        idx_results = i * nepisodes
        idx_gnorms = i * nepisodes


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
            if policy_plot is not None:
                def episode_dplot(sys, episode):
                    policy.plot_update()

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


    if processes > 1:  # NOTE parallelized
        with mp.Pool(processes=processes) as pool:
            result = pool.map(run, range(nruns))
    else:  # NOTE serialized
        for ri in range(nruns):
            run(ri)

    q_returns.put(None)
    q_gnorms.put(None)

    #  keeps figures alive
    import IPython
    IPython.embed()

import rl.pomdp as pomdp
import rl.pomdp.envs as envs
import rl.pomdp.policies as policies
import rl.pomdp.agents as agents
import rl.pomdp.callbacks as cbacks
import rl.misc as misc

import rl.values as value
import rl.values.v as v
import rl.values.q as q

import numpy as np
import numpy.random as rnd

import logging.config
from logconfig import LOGGING

import seaborn as sns
sns.set_style('darkgrid')

import multiprocessing as mp
import time


if __name__ == '__main__':
    # logging configuration
    logging.config.dictConfig(LOGGING)


    nruns, nepisodes, horizon = 100, 200, 50


    # .01
    # reactive:  ok;  listen then open
    # fsc(5):

    # .05
    # reactive:  not ok;  just listen
    # fsc(5):

    # NOTE initially, the nodes of the FSC are much less informative than the
    # nodes of the reactive policy........ perhaps use different step sizes?

    # # pomdp_fname = 'Tiger.pomdp'
    # pomdp_fname = 'loadunload.POMDP'
    # # pomdp_fname = 'heaven-hell.POMDP'
    # # pomdp_fname = 'Hallway.pomdp'
    # # pomdp_fname = 'Hallway2.pomdp'
    # # pomdp_fname = 'TagAvoid.pomdp'  # funny;  probabilities don't sum up to 1

    envname = 'Tiger'
    # envname = 'loadunload'
    # envname = 'heaven-hell'
    # envname = 'Hallway'
    # envname = 'Hallway2'
    # envname = 'TagAvoid'  # funny;  probabilities don't sum up to 1
    with envs.dotpomdp(envname) as f:
    # with open(pomdp_fname) as f:
        env = envs.parse(f)

    # with pomdp.f('loadunload') as f:
    #     pass

    # env = envs.Tiger(.15)
    # env.gamma = .95

    beta = .9

    # Random
    # policy = policies.Random(env)
    # agent = agents.Agent('Random', env, policy)

    # Blind (tries to learn best action distribution)
    # policy = policies.Blind(env)
    # agent = agents.Blind('Blind', env, policy)

    # GPOMDP
    # policy = policies.Reactive(env)
    # agent = agents.GPOMDP(f'GPOMDP ($\\beta={beta}$)', env, policy, beta)

    # Istate-GPOMDP (params N and beta)
    N = 5
    policy = policies.FSC(env, N)
    agent = agents.IsGPOMDP(f'IsGPOMDP (N={N}, $\\beta={beta}$)', env, policy, beta)

    plt_pdict = {
        100: dict(c='green', linestyle='dashed', linewidth=1, label='1.00'),
         75: dict(c='green', linewidth=1, label='0.75'),
         50: dict(c='black', label='0.50'),
         25: dict(c='red', linewidth=1, label='0.25'),
          0: dict(c='red', linestyle='dashed', linewidth=1, label='0.00'),
    }

    def plt_init():
        #  making sure pyplot is imported
        import matplotlib.pyplot as plt
        plt.ion()
        plt.title(f'Learning Performance ({agent.name})')
        plt.xlabel('Learning Episode')
        plt.ylabel('Expected Return $\mathbb{E}[ G_0 ]$')
        plt.legend(loc='lower right', frameon=True)
        # plt.legend(loc='lower center', frameon=True, ncol=np.inf, mode='expand')

    pplotter = cbacks.PLT_PercentilePlotter(
        env,
        (nruns, nepisodes),
        plt_pdict,
        plt_init=plt_init
    )
    callbacks = pplotter,

    horizon = misc.Horizon(horizon)
    sys = pomdp.System(env, env.model, horizon)

    v = mp.RawValue('i', 0)
    l = mp.Lock()
    def run(ri):
        with l:
            print(f'Starting run {v.value} / {nruns};  Running {nepisodes} episodes...')
            v.value += 1
        rnd.seed()  # ensure different randomization

        # setting plotting data index
        pplotter.idx = ri * nepisodes

        #  reset agent before each new learning run
        agent.reset()
        sys.run(
            agent,
            nepisodes=nepisodes,
            callbacks=callbacks,
        )

    # NOTE parallelized
    with mp.Pool() as pool:
        pool.map(run, range(nruns))

    # NOTE serialized
    # for ri in range(nruns):
    #     run(ri)

    # np.set_printoptions(precision=2, suppress=True)

    # this would kill the figures!
    # for cb in callbacks:
    #     cb.close()

    #  keeps figures alive
    import IPython
    IPython.embed()

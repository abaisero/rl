import rl.pomdp as pomdp
import rl.pomdp.envs as envs
import rl.pomdp.policies as policies
import rl.pomdp.psearch as psearch
import rl.pomdp.agents as agents
import rl.pomdp.callbacks as cbacks
import rl.optim as optim
import rl.misc as misc

# import rl.values as value
# import rl.values.v as v
# import rl.values.q as q

import numpy as np
import numpy.random as rnd

import logging.config
from logconfig import LOGGING

import seaborn as sns
sns.set_style('darkgrid')

import multiprocessing as mp
import time


if __name__ == '__main__':
    # np.seterr(all='raise')

    # logging configuration
    logging.config.dictConfig(LOGGING)

    # nruns, nepisodes, horizon = 100, 10000, 100
    nruns, nepisodes, horizon = 100, 10000, 100

    envname = 'Tiger'
    # envname = 'loadunload'
    # envname = 'heaven-hell'
    # envname = 'Hallway'
    # envname = 'Hallway2'
    # envname = 'TagAvoid'  # funny;  probabilities don't sum up to 1
    # with envs.dotpomdp(envname) as f:
    #     env = envs.parse(f)

    env = envs.Tiger(.01)
    env.gamma = .95

    N = 5
    beta = .95
    # step_size = optim.StepSize(10)
    step_size = optim.Geometric(10, .99)
    eps = 1e-10
    parall = False
    parall = True

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

    def pdict_item(p, **kwargs):
        return p, dict(label=f'{p/100:.2f}', **kwargs)

    pdict = dict([
        pdict_item(100, c='green', linestyle='dashed', linewidth=1),
        pdict_item( 75, c='green', linewidth=1),
        pdict_item( 50, c='black'),
        pdict_item( 25, c='red', linewidth=1),
        pdict_item(  0, c='red', linestyle='dashed', linewidth=1),
    ])

    def plt_init():
        #  making sure pyplot is imported
        import matplotlib.pyplot as plt
        plt.ion()
        plt.title(f'Learning Performance ({agent.name})')
        plt.xlabel('Learning Episode')
        plt.ylabel('Expected Return $\mathbb{E}[ G_0 ]$')
        plt.legend(loc='lower right', frameon=True)

    pplotter = cbacks.PLT_PercentilePlotter(
        env,
        (nruns, nepisodes),
        pdict,
        plt_init=plt_init
    )
    callbacks = pplotter,

    horizon = misc.Horizon(horizon)
    sys = pomdp.System(env, env.model, horizon)

    v = mp.RawValue('i', 0)
    l = mp.Lock()
    def run(ri):
        with l:
            # this index is a better one!! linear throughout processes
            i = v.value
            v.value += 1
        print(f'Starting run {i} / {nruns};  Running {nepisodes} episodes...')
        rnd.seed()  # ensure different randomization

        # setting plotting data index
        # pplotter.idx = ri * nepisodes
        pplotter.idx = i * nepisodes

        #  reset agent before each new learning run
        agent.reset()
        sys.run(
            agent,
            nepisodes=nepisodes,
            callbacks=callbacks,
        )

    if parall: # NOTE parallelized
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            pool.map(run, range(nruns))
    else: # NOTE serialized
        for ri in range(nruns):
            run(ri)

    # np.set_printoptions(precision=2, suppress=True)

    # this would kill the figures!
    # for cb in callbacks:
    #     cb.close()

    #  keeps figures alive
    import IPython
    IPython.embed()

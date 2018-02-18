import rl.mdp as mdp
import rl.mdp.envs as envs
import rl.mdp.policies as policies
import rl.mdp.agents as agents
import rl.misc as misc

import rl.values as values
import rl.values.v as v
import rl.values.q as q

import numpy as np
import numpy.random as rnd

import logging.config
from logconfig import LOGGING


if __name__ == '__main__':
    # logging configuration
    logging.config.dictConfig(LOGGING)


    nepisodes, horizon = 200, 20

    env = envs.Gridworld()
    env.gamma = .9


    # TODO extract "counts" stuff separately.. implement callbacks!



    # Monte Carlo Exploring Starts (evaluation)
    # Q = q.Tabular(env)
    # TODO I think init_value does not do anything right now..
    # # Q.init_value = 10  # helps early exploration
    # policy = policies.eGreedy(env, Q)
    # agent = agents.MonteCarloES('MC-ES', env, policy, Q)
    # V = v.QBased(Q, policy)

    # Monte Carlo Control
    # Q = q.Tabular(env)
    # # Q.init_value = 10  # helps early exploration
    # policy = policies.eGreedy(env, Q, .2)
    # agent = agents.MonteCarloControl('MC-Control', env, policy, Q)
    # V = v.QBased(Q, policy)

    # TD (evaluation)
    # V = v.Tabular(env)
    # policy = policies.Random(env)
    # agent = agents.TD('TD', env, policy, V)
    # Q = q.VBased(V, env.model, env.gamma)

    # SARSA (TD control)
    # Q = q.Tabular(env)
    # policy = policies.eGreedy(env, Q, e=.1)
    # agent = agents.SARSA('SARSA', env, policy, Q)
    # V = v.QBased(Q, policy)
    # C = agent.counts

    # Q-learning (TD off-policy control)
    # Q = q.Tabular(env)
    # policy = policies.eGreedy(env, Q, e=.1)
    # agent = agents.Qlearning('Qlearning', env, policy, Q)
    # V = v.QBased(Q, policy)
    # C = agent.counts

    # Expected SARSA
    # Q = q.Tabular(env)
    # policy = policies.eGreedy(env, Q, e=.1)
    # agent = agents.ExpectedSARSA('ExpectedSARSA', env, policy, Q)
    # V = v.QBased(Q, policy)

    # Double Q-learning (TD off-policy control, no maximization bias)
    # TODO this will be weird
    # Q1 = q.Tabular(env)
    # Q2 = q.Tabular(env)
    # Q = values.AvgValue((Q1, Q2))
    # policy = policies.eGreedy(env, Q, e=.1)
    # agent = agents.DoubleQlearning('Double-Qlearning', env, policy, Q1, Q2)
    # V = v.QBased(Q, policy)

    # TD(lambda) (evaluation)
    # V = v.Tabular(env)
    # # TODO policies as.... this stuff
    # policy = policies.Random(env)
    # # TODO how is alpha chosen in lambda methods?
    # agent = agents.TD_l('TD(l)', env, policy, V, gamma=env.gamma, l=0)
    # Q = q.VBased(V, env.model, env.gamma)

    # TODO add other feedbacks!!  two types.. episode feedback and just feedback

    # TODO agent callbacks!

    # TODO why is gamma given explicitly?  it's just inside env!

    # SARSA(lambda) (TD on-policy control)
    # V = v.Tabular(env)
    # policy = policies.Random(env)
    # agent = agents.SARSA_l('SARSA(l)', env, policy, V, gamma=env.gamma, l=0)
    # Q = q.VBased(V, env.model, env.gamma)

    # Qlearning(lambda) (TD off-policy control)
    # Q = q.Tabular(env)
    # policy = policies.Random(env)
    # agent = agents.Qlearning_l('Qlearning(l)', env, policy, Q, gamma=env.gamma, l=0)
    # V = v.QBased(Q, policy)

    # REINFORCE (policy gradient)
    policy = policies.PSoftmax(env)
    agent = agents.REINFORCE('REINFORCE', env, policy)

    # TODO this is a case where I would want a separate callback to do stuff like count!!! (because agent doesn't count himself..)

    # TODO FSC!!!

    horizon = misc.Horizon(horizon)
    sys = mdp.System(env, env.model, horizon)
    sys.run(agent, nepisodes=nepisodes)

    np.set_printoptions(precision=2, suppress=True)

    # As = np.empty((5, 5), dtype=object)
    # for s in env.states:
    #     As[s.value.pos] = policy.amax(s, rnd_=True).value[0]
    # print('Policy:')
    # print('------')
    # print(As)

    try:
        V
    except NameError:
        pass
    else:
        values = np.empty((5, 5))
        for s in env.states:
            values[s.value.pos] = V[s]
        print('Values:')
        print('------')
        print(values)

    actions = np.empty((5, 5), dtype=object)
    for s in env.states:
        # actions[s.value.pos] = Q.optim_action(s).value[0]
        actions[s.value.pos] = policy.amax(s).value[0]
    print('Policy:')
    print('------')
    print(actions)

    try:
        C
    except NameError:
        pass
    else:
        counts = np.empty((5, 5))
        for s in env.states:
            counts[s.value.pos] = C[s].sum()
        print('Counts:')
        print('------')
        print(counts)


    # print(env.model.r.asarray)

    # s0 = next(env.states)
    # a = next(env.actions)
    # s1 = next(env.states)
    # print(list(env.model.s1.dist(s0, a)))

    # print(env.model.r.sample(s0, a, s1))

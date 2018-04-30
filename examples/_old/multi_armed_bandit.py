import numpy as np
import numpy.random as rnd

from rl.problems import SAPair
from rl.problems.bandits.mab import GaussianBandit, MAB
from rl.values import Values_TabularCounted
from rl.policy.policy import Policy_UCB, Policy_egreedy
from rl.algo.bandits import BanditAgent

import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, name, policy, feedback):
        self.name = name
        self.policy = policy
        self.feedback = feedback

    def __str__(self):
        return self.name


def play_once(mab, agent, nrounds):
    rewardlist = np.empty(nrounds)
    optimlist = np.empty(nrounds)
    for ri in range(nrounds):
        actions = mab.actionlist

        b = agent.policy.sample_a(actions)
        r = b.sample_r()
        agent.feedback(b, r)

        rewardlist[ri] = r
        optimlist[ri] = b in mab.optimb

    return rewardlist, optimlist


def play_all(mab, agent_makers, nepisodes, nrounds):
    nagents = len(agent_makers)

    rewards = np.empty((nagents, nepisodes, nrounds))
    optims = np.empty((nagents, nepisodes, nrounds))
    for ei in range(nepisodes):
        print(f'episode {ei+1:4} / {nepisodes:4}')
        for ai, agent_maker in enumerate(agent_makers):
            agent = agent_maker()
            rewardlist, optimlist = play_once(mab, agent, nrounds)
            rewards[ai, ei, :] = rewardlist
            optims[ai, ei, :] = optimlist

    return rewards, optims


if __name__ == '__main__':
    bandits = [
        GaussianBandit(0, 2),
        GaussianBandit(1, 2),
        GaussianBandit(2, 2),
    ]
    mab = MAB(bandits)

    def agent_maker_egreedy(name, e):
        def agent_maker():
            A = Values_TabularCounted.A()
            policy = Policy_egreedy.A(A, e)
            return Agent(name, policy, A.update_target)
        return agent_maker

    def agent_maker_ucb(name):
        def agent_maker():
            A = Values_TabularCounted.A()
            policy = Policy_UCB.A(A.value_sa, A.confidence_sa)
            return Agent(name, policy, A.update_target)
        return agent_maker

    agent_makers = [
        agent_maker_egreedy('greedy', 0.),
        agent_maker_egreedy('.01-greedy', .01),
        agent_maker_egreedy('.1-greedy', .1),
        agent_maker_ucb('ucb'),
    ]
    agents = [agent_maker() for agent_maker in agent_makers]

    nepisodes = 500  # number of runs
    nrounds = 500  # number of rounds per run

    rewards, optims = play_all(mab, agent_makers, nepisodes, nrounds)
    rewards = rewards.mean(axis=1)  # average over all runs
    optims = optims.mean(axis=1)  # average over all runs


    ### plotting;  unimportant wrt the topic at hand

    ax = plt.subplot(211)
    for agent, reward in zip(agents, rewards):
        plt.plot(reward, label=agent.name)
    plt.xlabel('time')
    plt.ylabel('reward')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    ax = plt.subplot(212)
    for agent, optim in zip(agents, optims):
        plt.plot(optim, label=agent.name)
    plt.xlabel('time')
    plt.ylabel('optimality')
    plt.ylim([0, 1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    plt.show()

from __future__ import division

import numpy as np
from scipy.stats import norm

from rl.problems import SAPair
from rl.problems.bandits.mab import GaussianBandit, MAB
from rl.values import Values_TabularCounted
from rl.policy.policy import Policy_egreedy, Policy_softmax

import matplotlib.pyplot as plt


class Agent:
    def __init__(self, name, policy, feedback):
        self.name = name
        self.policy = policy
        self.feedback = feedback

    def __str__(self):
        return self.name


def play_once(mab, agent, nrounds):
    rewardlist = [None] * nrounds
    optimlist = [None] * nrounds
    for i in range(nrounds):
        actions = mab.actionlist

        b = agent.policy.sample_a(actions)
        r = b.sample_r()
        agent.feedback(b, r)

        rewardlist[i] = r
        optimlist[i] = b in mab.optimb

    return rewardlist, optimlist


def play_all(mab_maker, agent_makers, nepisodes, nrounds):
    nagents = len(agent_makers)

    rewards = np.empty((nagents, nepisodes, nrounds))
    optims = np.empty((nagents, nepisodes, nrounds))
    for ei in range(nepisodes):
        mab = mab_maker()
        print(f'episode {ei+1:4d} / {nepisodes}')
        for ai, agent_maker in enumerate(agent_makers):
            agent = agent_maker()
            rewardlist, optimlist = play_once(mab, agent, nrounds)
            rewards[ai, ei, :] = rewardlist
            optims[ai, ei, :] = optimlist

    return rewards, optims


if __name__ == '__main__':

    nbandits = 10
    def mab_maker():
        bandits = [GaussianBandit(norm.rvs(), 1)
            for i in range(nbandits)]
        return MAB(bandits)

    # def agents_maker():
    #     for e in 0, .1, .01:
    #         A = Values_TabularCounted.A()
    #         policy = Policy_egreedy.A(A, e=e)
    #         yield Agent(f'{e}-greedy', policy, A.update_target)

    #     for tau in .1, .3, 1:
    #         A = Values_TabularCounted.A()
    #         policy = Policy_softmax.A(A, tau=tau)
    #         yield Agent(f'softmax(tau={tau})', policy, A.update_target)

    def agent_maker_egreedy(name, e):
        def agent_maker():
            A = Values_TabularCounted.A()
            policy = Policy_egreedy.A(A, e)
            return Agent(name, policy, A.update_target)
        return agent_maker

    def agent_maker_softmax(name, tau):
        def agent_maker():
            A = Values_TabularCounted.A()
            policy = Policy_softmax.A(A, tau=tau)
            return Agent(name, policy, A.update_target)
        return agent_maker

    agent_makers = [
        agent_maker_egreedy('greedy', 0.),
        agent_maker_egreedy('.01-greedy', .01),
        agent_maker_egreedy('.1-greedy', .1),
        agent_maker_softmax('softmax(tau=.1)', .1),
        agent_maker_softmax('softmax(tau=.3)', .3),
        agent_maker_softmax('softmax(tau=1)', 1),
    ]
    agents = [agent_maker() for agent_maker in agent_makers]

    nepisodes = 2000  # number of runs
    nrounds = 1000  # number of rounds per run

    rewards, optims = play_all(mab_maker, agent_makers, nepisodes, nrounds)
    rewards = rewards.mean(axis=1)  # average over all runs  (but not over rounds)
    optims = optims.mean(axis=1)  # average over all runs  (but not over rounds)


    ### plotting;  unimportant wrt the topic at hand

    ax = plt.subplot(211)
    for agent, reward in zip(agents, rewards):
        plt.plot(reward, label=agent.name)
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.ylim([0, 1.6])
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
    ax.set_position([box.x0, box.y0, box.width * .6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    # plt.savefig('ex2.2.png')
    plt.show()

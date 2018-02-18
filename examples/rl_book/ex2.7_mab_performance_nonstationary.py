from __future__ import division

import numpy as np
from scipy.stats import norm

from rl.problems import SAPair
from rl.problems.bandits import Bandit
from rl.problems.bandits.mab import MAB
from rl.values import Values_TabularCounted
from rl.policy.policy import Policy_egreedy, Policy_softmax

import matplotlib.pyplot as plt


class GaussianBandit_Nonstationary(Bandit):
    def __init__(self, mu, sg):
        super(GaussianBandit_Nonstationary, self).__init__()
        self.mu = mu
        self.sg = sg

    @property
    def maxr(self):
        return abs(self.mu) + 3 * self.sg

    @property
    def Er(self):
        return self.mu

    def sample_r(self):
        return norm.rvs(self.mu, self.sg)

    def randomwalk(self):
        """ Bandit nonstationarity """
        self.mu += norm.rvs(0, .02)


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

        for b in mab.model.bandits:
            b.randomwalk()

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
        bandits = [GaussianBandit_Nonstationary(norm.rvs(), 1)
            for i in range(nbandits)]
        return MAB(bandits)

    def agent_maker_egreedy_alpha(name, e, alpha):
        def agent_maker():
            A = Values_TabularCounted.A(alpha=alpha)
            policy = Policy_egreedy.A(A, e)
            return Agent(name, policy, A.update_target)
        return agent_maker

    agent_makers = [
        agent_maker_egreedy_alpha('.1-greedy(alpha=1/(k+1))',
            .1, lambda k: 1/(k+1)),
        agent_maker_egreedy_alpha('.1-greedy(alpha=.1)',
            .1, lambda k: .1),
    ]
    agents = [agent_maker() for agent_maker in agent_makers]

    nepisodes = 2000  # number of runs
    nrounds = 2000  # number of rounds per run

    rewards, optims = play_all(mab_maker, agent_makers, nepisodes, nrounds)
    rewards = rewards.mean(axis=1)  # average over all runs  (but not over rounds)
    optims = optims.mean(axis=1)  # average over all runs  (but not over rounds)


    ### plotting;  unimportant wrt the topic at hand
    # TODO also plot the performance of the same agents in stationary mab system
    # TODO the performance doesn't seem to be suffering from the non-stationarity.. investigate

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
    ax.set_position([box.x0, box.y0, box.width * .6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    # plt.savefig('ex2.7.png')
    plt.show()

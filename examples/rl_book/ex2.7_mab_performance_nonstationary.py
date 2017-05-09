from __future__ import division

import numpy as np
from scipy.stats import norm

from rl.problems import SAPair
from rl.problems.bandits import Bandit
from rl.problems.bandits.mab import MAB
from rl.values import Values_TabularCounted
from rl.policy import Policy_egreedy, Policy_softmax

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


class Agent(object):
    def __init__(self, name, policy_maker):
        self.name = name
        self.policy_maker = policy_maker
        self.reset()

    def reset(self):
        self.policy, self.update_target = self.policy_maker()

    def feedback(self, a, r):
        self.update_target(a, r)

    def __str__(self):
        return self.name


def play_once(mab, agent, nrounds):
    rewardlist = [None] * nrounds
    optimlist = [None] * nrounds
    for i in xrange(nrounds):
        actions = mab.actionlist

        b = agent.policy.sample_a(actions)
        r = b.sample_r()
        agent.feedback(b, r)

        rewardlist[i] = r
        optimlist[i] = b in mab.optimb

        for b in mab.model.bandits:
            b.randomwalk()

    return rewardlist, optimlist


def play_all(mab_maker, agents, nepisodes, nrounds):
    nagents = len(agents)

    rewards = np.empty((nagents, nepisodes, nrounds))
    optims = np.empty((nagents, nepisodes, nrounds))
    for ei in xrange(nepisodes):
        mab = mab_maker()
        print 'episode {:4d} / {}'.format(ei, nepisodes)
        for ai, agent in enumerate(agents):
            agent.reset()
            rewardlist, optimlist = play_once(mab, agent, nrounds)
            rewards[ai, ei, :] = rewardlist
            optims[ai, ei, :] = optimlist

    return rewards, optims


if __name__ == '__main__':

    nbandits = 10
    def mab_maker():
        bandits = [GaussianBandit_Nonstationary(norm.rvs(), 1) for i in xrange(nbandits)]
        return MAB(bandits)

    def policy_maker_egreedy(alpha, e):
        def policy_maker():
            A = Values_TabularCounted.A(alpha=alpha)
            policy = Policy_egreedy.A(A, e)
            return policy, A.update_target
        return policy_maker

    agents = [
        Agent('.1-greedy(alpha=1/(k+1))', policy_maker_egreedy(lambda n: 1/(n+1), .1)),
        Agent('.1-greedy(alpha=.1)', policy_maker_egreedy(lambda n: .1, .1)),
    ]

    nepisodes = 2000  # number of runs
    nrounds = 2000  # number of rounds per run

    rewards, optims = play_all(mab_maker, agents, nepisodes, nrounds)
    rewards = np.mean(rewards, axis=1)  # average over all runs  (but not over rounds)
    optims = np.mean(optims, axis=1)  # average over all runs  (but not over rounds)


    ### plotting;  unimportant wrt the topic at hand
    # TODO also plot the performance of the same agents in stationary mab system
    # TODO the performance doesn't seem to be suffering from the non-stationarity.. investigate

    ax = plt.subplot(211)
    for agent, reward in zip(agents, rewards):
        plt.plot(reward, label=agent.name)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    ax = plt.subplot(212)
    for agent, optim in zip(agents, optims):
        plt.plot(optim, label=agent.name)
    plt.ylim([0, 1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    plt.savefig('ex2.7.png')
    plt.show()

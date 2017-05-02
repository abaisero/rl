from __future__ import division

import numpy as np
from scipy.stats import norm

from rl.problems import SAPair
from rl.problems.bandits.mab import GaussianBandit, MAB
from rl.values import ActionValues_TabularCounted
from rl.policy import Policy_egreedy, Policy_softmax

import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, name, policy_cls, *policy_args, **policy_kwargs):
        self.name = name
        self.policy_cls = policy_cls
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs
        self.reset()

    def reset(self):
        self.policy = self.policy_cls(
            ActionValues_TabularCounted(),
            *self.policy_args,
            **self.policy_kwargs
        )

    def feedback(self, a, r):
        self.policy.Q.update_target(SAPair(a=a), r)

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
        bandits = [GaussianBandit(norm.rvs(), 1) for i in xrange(nbandits)]
        return MAB(bandits)

    agents = [
        Agent('greedy', Policy_egreedy, 0),
        Agent('.1-greedy', Policy_egreedy, .1),
        Agent('.01-greedy', Policy_egreedy, .01),
        Agent('softmax(tau=.1)', Policy_softmax, .1),
        Agent('softmax(tau=.3)', Policy_softmax, .3),
        Agent('softmax(tau=1)', Policy_softmax, 1),
    ]

    nepisodes = 2000  # number of runs
    nrounds = 1000  # number of rounds per run

    rewards, optims = play_all(mab_maker, agents, nepisodes, nrounds)
    rewards = np.mean(rewards, axis=1)  # average over all runs  (but not over rounds)
    optims = np.mean(optims, axis=1)  # average over all runs  (but not over rounds)


    ### plotting;  unimportant wrt the topic at hand

    ax = plt.subplot(211)
    for agent, reward in zip(agents, rewards):
        plt.plot(reward, label=agent.name)
    plt.ylim([0, 1.6])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    ax = plt.subplot(212)
    for agent, optim in zip(agents, optims):
        plt.plot(optim, label=agent.name)
    plt.ylim([0, 1])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    # plt.savefig('ex2.2.png')
    plt.show()

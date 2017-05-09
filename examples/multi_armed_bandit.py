import numpy as np
import numpy.random as rnd

from rl.problems import SAPair
from rl.problems.bandits.mab import GaussianBandit, MAB
from rl.values import Values_TabularCounted
from rl.policy import Policy_UCB, Policy_egreedy
from rl.algo.bandits import BanditAgent

import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, name, policy_maker):
        self.name = name
        self.policy_maker = policy_maker
        self.reset()

    def reset(self):
        self.policy, self.update_target = self.policy_maker()

    def feedback(self, b, r):
        self.update_target(b, r)

    def __str__(self):
        return self.name


def play_once(mab, agent, nrounds):
    rewardlist = np.empty(nrounds)
    optimlist = np.empty(nrounds)
    for ri in xrange(nrounds):
        actions = mab.actionlist

        b = agent.policy.sample_a(actions)
        r = b.sample_r()
        agent.feedback(b, r)

        rewardlist[ri] = r
        optimlist[ri] = b in mab.optimb

    return rewardlist, optimlist


def play_all(mab, agents, nepisodes, nrounds):
    nagents = len(agents)

    rewards = np.empty((nagents, nepisodes, nrounds))
    optims = np.empty((nagents, nepisodes, nrounds))
    for ei in xrange(nepisodes):
        print 'episode {:4} / {:4}'.format(ei, nepisodes)
        for ai, agent in enumerate(agents):
            agent.reset()
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

    def policy_maker_egreedy(e):
        def policy_maker():
            A = Values_TabularCounted.A()
            policy = Policy_egreedy.A(A, e)
            return policy, A.update_target
        return policy_maker

    def policy_maker_ucb():
        def policy_maker():
            A = Values_TabularCounted.A()
            policy = Policy_UCB.A(A.value_sa, A.confidence_sa)
            return policy, A.update_target
        return policy_maker

    agents = [
        Agent('greedy', policy_maker_egreedy(0.)),
        Agent('.01-greedy', policy_maker_egreedy(.01)),
        Agent('.1-greedy', policy_maker_egreedy(.1)),
        Agent('ucb', policy_maker_ucb()),
    ]

    nepisodes = 500  # number of runs
    nrounds = 500  # number of rounds per run

    rewards, optims = play_all(mab, agents, nepisodes, nrounds)
    rewards = np.mean(rewards, axis=1)  # average over all runs
    optims = np.mean(optims, axis=1)  # average over all runs


    ### plotting;  unimportant wrt the topic at hand

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
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, .5), fancybox=True, shadow=True)

    plt.show()

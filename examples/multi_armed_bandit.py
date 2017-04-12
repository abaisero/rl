import numpy.random as rnd

from rl.problems.mab import GaussianBandit, MAB
from rl.values import Values_Tabular
from rl.policy import Policy_UCB, UCB_confidence_Q
from rl.algo.bandits import BanditAgent


if __name__ == '__main__':
    bandits = [
        GaussianBandit(0, 2),
        GaussianBandit(1, 2),
        GaussianBandit(2, 2),
    ]
    mab = MAB(bandits)

    Q = Values_Tabular()
    confidence = lambda sa: UCB_confidence_Q(sa, Q)
    policy = Policy_UCB(Q.value, confidence, beta=2.)
    agent = BanditAgent(mab, Q, policy)

    for i in range(1000):
        a = agent.sample_a()
        r = mab.sample_r(a)

        # TODO plot real bandit distributions, and samples, and belief distributions
        # plt.waitforbuttonpress()

        print '{} {}'.format(a, r)
        agent.feedback(a, r)

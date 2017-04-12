import numpy.random as rnd

from rl.problems.mab import GaussianBandit, MAB
from rl.values import Values_Table
from rl.policy import Policy_UCB
from rl.algo.bandits import BanditAgent


if __name__ == '__main__':
    bandits = [
        GaussianBandit(0, 2),
        GaussianBandit(1, 2),
        GaussianBandit(2, 2),
    ]
    mab = MAB(bandits)

    Q = Values_Table()
    policy = Policy_UCB(Q.value, Q.UCB_confidence, beta=2.)
    agent = BanditAgent(mab, Q, policy)

    for i in range(100):
        a = agent.sample_a()
        r = mab.sample_r(a)

        # TODO plot real bandit distributions, and samples, and belief distributions
        # plt.waitforbuttonpress()

        print '{} {}'.format(a, r)
        agent.feedback(a, r)

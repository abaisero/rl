import numpy.random as rnd

from rl.problems import SAPair
from rl.problems.bandits.mab import GaussianBandit, MAB
from rl.values import Values_TabularCounted
from rl.policy import Policy_UCB, UCB_confidence_Q
from rl.algo.bandits import BanditAgent


if __name__ == '__main__':
    bandits = [
        GaussianBandit(0, 2),
        GaussianBandit(1, 2),
        GaussianBandit(2, 2),
    ]
    mab = MAB(bandits)

    Q = Values_TabularCounted()
    def Q_confidence(sa): return UCB_confidence_Q(sa, Q)
    policy = Policy_UCB(Q.value, Q_confidence, beta=mab.maxr)
    # policy = Policy_UCB(Q.value, Q_confidence, beta=1.)
    # agent = BanditAgent(cb, Q, policy)

    for i in range(1000):
        b = policy.sample_a(mab.actionlist)
        r = b.sample_r()
        # r = mab.model.sample_r(b)

        # TODO plot real bandit distributions, and samples, and belief distributions
        # plt.waitforbuttonpress()

        if i % 100 == 0:
            print '{} {}'.format(b, r)
        # NOTE I have to use SAPair because policy uses SAPair.
        Q.update_target(SAPair(a=b), r)
        # agent.feedback(a, r)

    print
    print 'Final values'
    print '###'
    for b in bandits:
        sa = SAPair(a=b)
        print '{} {} {}'.format(b, Q.value(sa), Q.nupdates_sa(sa))

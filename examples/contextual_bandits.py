import numpy.random as rnd

from rl.problems import State, SAPair
from rl.problems.bandits.cb import ContextualBandit, CBModel, CB
from rl.values import Values_TabularCounted
from rl.policy import Policy_UCB, UCB_confidence_Q
from rl.algo.bandits import BanditAgent


def roll():
    return rnd.choice(6) + 1


class DiceState(State):
    def __init__(self, npips):
        self.setkey((npips,))

        self.npips = npips

    def __str__(self):
        return 'S(npips={})'.format(self.npips)


class DiceModel(CBModel):
    def sample_s0(self):
        return DiceState(roll())


class DiceBandit(ContextualBandit):
    def __init__(self, sign):
        self.sign = cmp(sign, 0)

        self.maxr = 1

    def sample_r(self, s):
        # import ipdb;  ipdb.set_trace()
        return int(cmp(roll(), s.npips) == self.sign)

    def __str__(self):
        sign = '+' if self.sign > 0 else '-' if self.sign < 0 else '='
        return 'B({})'.format(sign)


if __name__ == '__main__':
    bandits = [
        DiceBandit(sign=1),
        DiceBandit(sign=0),
        DiceBandit(sign=-1),
    ]
    model = DiceModel(bandits)
    cb = CB(model)

    Q = Values_TabularCounted()
    # def Q_confidence(sa): return UCB_confidence_Q(sa, Q)
    # policy = Policy_UCB(Q.value, Q_confidence, beta=cb.maxr)
    policy = Policy_UCB(Q.value, lambda sa: UCB_confidence_Q(sa, Q), beta=cb.maxr)
    # agent = BanditAgent(mab, Q, policy)

    for i in range(100000):
        s = cb.model.sample_s0()
        b = policy.sample_a(cb.actionlist)
        r = b.sample_r(s)

        # TODO plot real bandit distributions, and samples, and belief distributions
        # plt.waitforbuttonpress()

        if i % 1000 == 0:
            print '{} {} {}'.format(s, b, r)
        Q.update_target(SAPair(s, b), r)
        # agent.feedback(SAPair(s, b), r)

    print
    print 'Final Values'
    print '###'
    for i in range(1, 7):
        s = DiceState(i)
        for b in bandits:
            sa = SAPair(s, b)
            print '{} {}: {} {}'.format(s, b, Q.value(sa), Q.nupdates_sa(sa))

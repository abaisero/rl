import numpy.random as rnd

from rl.problems import State, SAPair
from rl.problems.bandits.cb import ContextualBandit, CBModel, CB
from rl.values import Values_TabularCounted
from rl.policy.policy import Policy_UCB

import indextools


nsides = 6
def roll():
    return rnd.choice(nsides) + 1


values = list(range(1, nsides+1))
sspace = indextools.DomainSpace(values)


class DiceModel(CBModel):
    def sample_s0(self):
        return sspace.item(value=roll())
        # return DiceState(roll())


# python3 doesn't have cmp
def cmp(a, b):
    return ((a>b) - (a<b))


class DiceBandit(ContextualBandit):
    def __init__(self, sign):
        self.setkey((sign,))
        self.sign = cmp(sign, 0)

        self.maxr = 1

    def sample_r(self, s):
        return int(cmp(roll(), s.value) == self.sign)

    def __str__(self):
        sign = ('+' if self.sign > 0 else
                '-' if self.sign < 0 else '=')
        return f'B({sign})'


if __name__ == '__main__':
    bandits = [
        DiceBandit(sign=1),
        DiceBandit(sign=0),
        DiceBandit(sign=-1),
    ]
    model = DiceModel(bandits)
    cb = CB(model)

    Q = Values_TabularCounted.Q()
    policy = Policy_UCB.Q(Q.value, Q.confidence, beta=cb.maxr)

    for i in range(100000):
        s = cb.model.sample_s0()
        b = policy.sample(s, cb.actionlist)
        r = b.sample_r(s)

        # TODO plot real bandit distributions, and samples, and belief distributions
        # plt.waitforbuttonpress()

        if i % 1000 == 0:
            print(f'{s} {b} {r}')
        Q.update_target(s, b, r)

    print()
    print('final values')
    for npips in values:
        s = sspace.item(value=npips)
        for b in bandits:
            print(f'{s} {b}: {Q(s, b):.2f} {Q.confidence(s, b):.2f}')

    print()
    print('optim actions')
    for npips in values:
        s = sspace.item(value=npips)
        print(f'{s}: {Q.optim_action(s, cb.actionlist)}')

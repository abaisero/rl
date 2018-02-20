# TODO version one here

class Bandit(object):
    def sample_r(self):
        raise NotImplementedError


class BinaryBandit(Bandit):
    def __init__(self, p):
        super(BinaryBandit, self).__init__()
        # if not 0 <= p <= 1:
        #     raise BanditException('BinaryBandit parameter is not in interval [0, 1]. (given {}).'.format(p))
        self.p = p

        self.maxr = 1
        self.Er = p

    def sample_r(self):
        return int(rnd.uniform() < self.p)


class GaussianBandit(Bandit):
    def __init__(self, m, s):
        super(GaussianBandit, self).__init__()
        self.m = m
        self.s = s
        self.rv = norm(m, s)

        self.maxr = max(abs(m + 3 * s), abs(m - 3 * s))
        self.Er = m

    def sample_r(self):
        return self.rv.rvs()


class ChoiceBandit(Bandit):
    def __init__(self, choices):
        super(ChoiceBandit, self).__init__()
        self.choices = choices
        self.nchoices = len(choices)

        self.maxr = max(map(abs, choices))
        self.Er = np.mean(choices)

    def sample_r(self):
        ci = rnd.choice(self.nchoices)
        return self.choices[ci]

# TODO version two here

# from .. import Action


class BanditException(Exception):
    pass


class Bandit(Action):
    def __init__(self):
        self.setkey(('bidx',), names=True)

    def sample_r(self):
        raise NotImplementedError

    def __str__(self):
        return 'B({})'.format(self.bidx)


class ContextualBandit(Bandit):
    def sample_r(self, s):
        raise NotImplementedError

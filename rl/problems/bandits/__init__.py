from .. import Action


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

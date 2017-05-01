from .. import Action


class BanditException(Exception):
    pass


class Bandit(Action):
    # def __init__(self):
    #     Bandit.reset_nbandits()
    #     self.bidx = Bandit.nbandits
    #     Bandit.nbandits += 1

    # @classmethod
    # def reset_nbandits(cls):
    #     cls.nbandits = 0

    def sample_r(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.bidx)

    def __eq__(self, other):
        return self.bidx == other.bidx

    def __str__(self):
        return 'B({})'.format(self.bidx)


class ContextualBandit(Bandit):
    def sample_r(self, s):
        raise NotImplementedError


# TODO in bandit problems, bandits ARE actions
class BanditAction(Action):
    def __init__(self, ai):
        self.ai = ai

    def __hash__(self):
        return hash(self.ai)

    def __eq__(self, other):
        return self.ai == other.ai

    def __str__(self):
        return 'A({})'.format(self.ai)

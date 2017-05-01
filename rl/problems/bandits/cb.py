from .. import State, Model, RLProblem
from . import ContextualBandit


class CBModel(Model):
    def __init__(self, bandits):
        for bidx, b in enumerate(bandits):
            b.bidx = bidx
        self.bandits = bandits
        # self.nbandits = len(bandits)

    def sample_s0(self):
        pass

    def sample_r(self, sb):
        return sb.a.sample_r(sb.s)


class CB(RLProblem):
    def __init__(self, model):
        super(CB, self).__init__(model)
        self.actionlist = model.bandits

        self.maxr = max(b.maxr for b in model.bandits)

    # def sample_r(self, sb):
    #     return self.model.sample_r(sa)

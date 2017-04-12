from . import Model, RLProblem
# TODO Contextual Bandits classes


class CBModel(Model):
    def __init__(self, bandits):
        self.bandits = bandits
        self.nbandits = len(bandits)

    def sample_r(self, s, a):
        return self.bandits[a].sample_r(s)


class CB(RLProblem):
    def __init__(self, model):
        super(CB, self).__init__(model)
        self.actionlist = range(self.model.nbandits)

    def sample_r(self, s, a):
        return self.model.sample_r(s, a)

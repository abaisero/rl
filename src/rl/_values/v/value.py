import rl.values as rlvalues


class V(rlvalues.Value):
    def value(self, s):
        return self[s]

    def __getitem__(self, sk):
        return self.vmodel[sk]

    def __setitem__(self, sk, value):
        self.vmodel[sk] = value


class Counts(rlvalues.Counts):
    def reset(self):
        self.cmodel[:] = 0

    def count(self, s):
        return self[s]

    def __getitem__(self, sk):
        return self.cmodel[sk]

    def __setitem__(self, sk, value):
        self.cmodel[sk] = value


class Eligibility(rlvalues.Eligibility):
    def reset(self):
        self.emodel[:] = 0.

    def elig(self, s):
        return self[s]

    def __getitem__(self, sk):
        return self.emodel[sk]

    def __setitem__(self, sk, value):
        self.emodel[sk] = value

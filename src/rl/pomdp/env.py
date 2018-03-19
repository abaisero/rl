import rl.mdp as mdp


class Environment(mdp.Environment):
    def __init__(self, sspace, aspace, ospace):
        super().__init__(sspace, aspace)
        self.ospace = ospace

    @property
    def obs(self):
        return self.ospace.elems

    @property
    def nobs(self):
        return self.ospace.nelems

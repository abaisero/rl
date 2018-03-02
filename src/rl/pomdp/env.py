import rl.mdp as mdp


class Environment(mdp.Environment):
    def __init__(self, sfactory, afactory, ofactory):
        super().__init__(sfactory, afactory)
        self.ofactory = ofactory

    @property
    def obs(self):
        return self.ofactory.items

    @property
    def nobs(self):
        return self.ofactory.nitems

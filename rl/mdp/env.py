# TODO make StateSpace ActionSpace and ObsSpace classes;  which essentially
# contains other factories..

# NOTE existence of tfactory determines whether this is an episodic environment
# or not

class Environment:
    def __init__(self, sfactory, afactory, tfactory=None):
        self.sfactory = sfactory
        self.afactory = afactory
        # TODO how to handle distributions?  joint / union space?
        self.tfactory = tfactory

        self.episodic = tfactory is not None
        self.continuous = tfactory is None

    def isterminal(self, state):
        return self.tfactory.isitem(state)

    @property
    def states(self):
        return self.sfactory.items

    @property
    def nstates(self):
        return self.sfactory.nitems

    @property
    def tstates(self):
        return self.tfactory.items

    @property
    def ntstates(self):
        return self.tfactory.nitems

    @property
    def actions(self):
        return self.afactory.items

    @property
    def nactions(self):
        return self.afactory.nitems

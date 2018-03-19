# TODO make StateSpace ActionSpace and ObsSpace classes;  which essentially
# contains other factories..

# NOTE existence of tfactory determines whether this is an episodic environment
# or not

class Environment:
    def __init__(self, sspace, aspace, tspace=None):
        self.sspace = sspace
        self.aspace = aspace
        # TODO how to handle distributions?  joint / union space?
        self.tspace = tspace

        self.episodic = tspace is not None
        self.continuous = tspace is None

    def isterminal(self, state):
        return self.tspace.isitem(state)

    @property
    def states(self):
        return self.sspace.elems

    @property
    def nstates(self):
        return self.sspace.nelems

    @property
    def tstates(self):
        return self.tspace.elems

    @property
    def ntstates(self):
        return self.tspace.nelems

    @property
    def actions(self):
        return self.aspace.elems

    @property
    def nactions(self):
        return self.aspace.nelems

class Domain:
    def __repr__(self):
        return f'Domain(|S|={self.nstates}, |A|={self.nactions})'

    def __init__(self, sspace, aspace, model=None, gamma=None):
        self.sspace = sspace
        self.aspace = aspace
        self.model = model
        self.gamma = gamma

    @property
    def states(self):
        return self.sspace.elems

    @property
    def nstates(self):
        return self.sspace.nelems

    @property
    def actions(self):
        return self.aspace.elems

    @property
    def nactions(self):
        return self.aspace.nelems

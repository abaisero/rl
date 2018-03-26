class Domain:
    def __repr__(self):
        return f'Domain(|S|={self.nstates}, |A|={self.nactions}, |O|={self.nobs})'

    def __init__(self, sspace, aspace, ospace, model=None, gamma=None):
        self.sspace = sspace
        self.aspace = aspace
        self.ospace = ospace
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

    @property
    def obs(self):
        return self.ospace.elems

    @property
    def nobs(self):
        return self.ospace.nelems

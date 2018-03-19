# TODO this is not nice....  what does this represent..?
class Distribution:
    def __init__(self, *, cond=None):
        if cond is None:
            cond = ()
        xspaces = cond

        self.xspaces = xspaces
        self.nx = len(xspaces)

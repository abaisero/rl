class FModel:
    def __init__(self, factories):
        self.factories = factories

    def __getitem__(self, items):
        raise NotImplementedError

    def __setitem__(self, items, value):
        raise NotImplementedError

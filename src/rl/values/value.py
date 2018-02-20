class Value:
    def __init__(self, env):
        self.env = env

    def value(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


    # def confidence(self, *args, **kwargs):
    #     raise NotImplementedError

    # def update_value(self, *args, **kwargs):
    #     raise NotImplementedError

    # def update_target(self, *args, **kwargs):
    #     raise NotImplementedError

    # def __getitem__(self, k):
    #     return self.value(*k)

    # def __setitem__(self, k, value):
    #     self.update_value(*k, value)


class Counts:
    def count(self, *args, **kwargs):
        raise NotImplementedError


class Eligibility:
    def __init__(self, gamma, l):
        self.gl = gamma * l

    def elig(self, *args, **kwargs):
        raise NotImplementedError

    # def update_elig(self, *args, **kwargs):
    #     raise NotImplementedError

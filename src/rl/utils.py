from math import exp, log


class ExpDecay:
    def __init__(self, a, b, n):
        self.a = a
        self.b = b
        self.n = n

        self.l = (log(b) - log(a)) / n
        self.reset()

    def reset(self):
        self.i = 0

    def __call__(self):
        return self.a * exp(self.l * self.i)

    def step(self):
        v = self()
        self.i += 1
        return v


class NoDecay:
    def __init__(self, v):
        self.v = v

    def reset(self):
        pass

    def __call__(self):
        return self.v

    def step(self):
        return self.v


import argparse


class DecayAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        print(values)
        # setattr(args, self.dest, value)


class wtuple:
    def __init__(self, *args):
        self.__objs = args

    def __getitem__(self, key):
        return self.__objs[key]

    def __getattr__(self, attr):
        def f(*args, **kwargs):
            objs = [getattr(t, attr)(*args, **kwargs)
                    for t in self.__objs]
            return wtuple(*objs)
        return f

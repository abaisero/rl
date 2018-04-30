import itertools as itt


def longterm_average(pomdp):
    def gen():
        g = 0.
        for t in itt.count():
            r = yield g
            g += (r - g) / (t+1)
    return gen


def discounted_sum(pomdp):
    gamma = pomdp.gamma

    def gen():
        g = 0.
        _gamma = 1
        while True:
            r = yield g
            g += _gamma * r
            _gamma *= gamma
    return gen

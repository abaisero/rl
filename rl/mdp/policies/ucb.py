from .policy import Policy


class UCB(Policy):
    def __init__(self, env, mu, sg, beta):
        super().__init__(env)
        self.mu = mu
        self.sg = sg
        self.beta = beta

    def ucb(self, s, a):
        return self.mu(s, a) + self.beta * np.nan_to_num(self.sg(s, a))

    def dist(self, s):
        amax = argmax(
            functools.partial(self, ucb, s),
            self.env.actions,
            all_=True
        )
        pramax = 1 / len(amax)
        for a in amax:
            yield a, pramax

    def sample(self, s):
        return = argmax(
            functools.partial(self, ucb, s),
            self.env.actions,
            rnd_=True
        )

from rl.algo.episode import run_episode, make_returns


class MC(object):
    def __init__(self, sys, model, policy, Q):
        self.sys = sys
        self.model = model
        self.policy = policy
        self.Q = Q

    def run(self, s, verbose=False):
        first_visit = True

        episode = run_episode(self.sys, self.model, self.policy, s, verbose=verbose)
        returns = make_returns(episode, self.model.gamma, first_visit=first_visit)

        if first_visit:
            for (s, a), R in returns.viewitems():
                Q.update_target(s, a, R)
        else:
            for (s, a), Rl in returns.viewitems():
                for R in Rl:
                    Q.update_target(s, a, R)

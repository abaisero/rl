from rl.algo.episode import run_episode, make_returns
from . import Algo


class MC(Algo):
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

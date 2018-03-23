import rl.mdp as mdp


class Environment(mdp.Environment):
    def __init__(self, sspace, aspace, ospace):
        super().__init__(sspace, aspace)
        self.ospace = ospace

    @property
    def obs(self):
        return self.ospace.elems

    @property
    def nobs(self):
        return self.ospace.nelems

    def discounted_sum(self, episode):
        """J \eqdot \sum_{t=1}^T r_t \lambda^t"""
        G = 0.
        for _, _, feedback, _ in episode:
            G = self.gamma * G + feedback.r
        return G

    @staticmethod
    def longterm_average(episode):
        r"""\nu \eqdot \frac{1}{T} \sum_{t=1}^T r_t"""
        return sum(feedback.r for _, _, feedback, _ in episode) / len(episode)

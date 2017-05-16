from . import Algo


class Qlearning(Algo):
    def run(self, s, verbose=False):
        actions = self.sys.actions(s)

        if verbose:
            print '---'

        while not s.terminal:
            a = self.policy.sample(s, actions)
            r, s1 = self.model.sample_rs1(s, a)

            if verbose:
                print '{}, {}, {}, {}'.format(s, a, r, s1)

            actions1 = self.sys.actions(s1)
            target = r + self.model.task.gamma * self.Q.optim_value(s1, actions1)
            self.Q.update_target(s, a, target)

            s, actions = s1, actions1


class Qlearning_l(Algo):
    def __init__(self, sys, model, policy, Q, lambda_):
        # TODO implement Watkin's Q
        raise Exception('Watkin\'s Q is still not implemented')

        super(Qlearning_l, self).__init__(sys, model, policy, Q)
        self.lambda_ = lambda_

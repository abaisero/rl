from . import Algo


class Qlearning(Algo):
    def run(self, s, verbose=False):
        actions = self.sys.actions
        # actions = self.sys.actions(s)

        if verbose:
            print('---')

        terminal = False
        # while not s.terminal:
        while not terminal:
            a = self.policy.sample(s, self.sys.actions)
            # a = self.policy.sample(s, actions)
            s1, terminal = self.model.s1.sample_s1(s, a)
            r = self.model.r.sample_r(s, a, s1)
            # r, s1 = self.model.sample_rs1(s, a)

            if verbose:
                print(f'{s.value}, {a.value}, {r}, {s1.value}, {terminal}')

            # actions1 = self.sys.actions(s1)
            actions1 = self.sys.actions
            target = r + self.model.gamma * self.Q.optim_value(s1, actions1)
            self.Q.update_target(s, a, target)

            s, actions = s1, actions1


class Qlearning_l(Algo):
    def __init__(self, sys, model, policy, Q, lambda_):
        # TODO implement Watkin's Q
        raise Exception('Watkin\'s Q is still not implemented')

        super(Qlearning_l, self).__init__(sys, model, policy, Q)
        self.lambda_ = lambda_

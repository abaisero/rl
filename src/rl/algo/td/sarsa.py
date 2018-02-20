from . import Algo


class SARSA(Algo):
    def run(self, s, verbose):
        actions = self.sys.actions(s)
        a = self.policy.sample(s, actions)

        if verbose:
            print('---')

        while not s.terminal:
            s1 = self.model.s1.sample(s, a)
            r = self.model.r.sample(s, a, s1)
            # r, s1 = self.model.sample_rs1(s, a)
            actions1 = self.sys.actions(s1)
            a1 = self.policy.sample(s1, actions1)

            if verbose:
                print(f'{s}, {a}, {r}, {s1}, {a1}')

            target = r + self.model.gamma * self.Q(s1, a1)
            self.Q.update_target(s, a, target)

            s, a = s1, a1


class SARSA_l(Algo):
    def __init__(self, sys, model, policy, Q, lambda_):
        super(SARSA_l, self).__init__(sys, model, policy, Q)
        self.lambda_ = lambda_

    def run(self, s, verbose=False):
        actions = self.sys.actions(s)
        a = self.policy.sample(s, actions)

        if verbose:
            print('---')

        with self.Q.eligibility(self.model.gamma, self.lambda_) as elig:  # interesting;; this creates an eligibility trace structure.....
            while not s.terminal:
                s1 = self.model.s1.sample(s, a)
                r = self.model.r.sample(s, a, s1)
                # r, s1 = self.model.sample_rs1(s, a)
                actions1 = self.sys.actions(s1)
                a1 = self.policy.sample(s1, actions1)

                if verbose:
                    print(f'{s}, {a}, {r}, {s1}, {a1}')

                elig.update(s, a)
                target = r + self.model.gamma * self.Q(s1, a1)
                self.Q.update_target(s, a, target)

                s, a = s1, a1

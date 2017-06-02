class Qlearning(object):
    def __init__(self, sys, policy, Q, Q_tasks=None):
        if Q_tasks is None:
            Q_tasks = []
        self.sys = sys
        self.policy = policy
        self.Q_tasks = [(Q, sys.model.task)] + Q_tasks
        self.Qs = [Q for Q, _ in self.Q_tasks]

    def run(self, s, callback_step=None, verbose=False):
        actions = self.sys.actions(s)

        if verbose:
            print '---'

        # i = 0
        while not self.sys.model.task.is_terminal(s):
            # if i == 500:
            #     break
            # i += 1
            a = self.policy.sample(s, actions)
            s1 = self.sys.model.dynamics.sample_s1(s, a)
            actions1 = self.sys.actions(s1)

            verbose_ = verbose
            for Q, task in self.Q_tasks:
                r = task.sample_r(s, a, s1)

                if verbose_:
                    print '{}, {}, {}, {}'.format(s, a, r, s1)
                    verbose_ = False

                if not task.is_terminal(s1):
                    target = r + task.gamma * Q.optim_value(s1, actions1)
                    Q.update_target(s, a, target)

            if callback_step is not None:
                callback_step(s0=s, a=a, s1=s1, Qs=self.Qs)

            s, actions = s1, actions1

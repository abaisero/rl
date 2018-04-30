import rl.values as rlvalues


class A(rlvalues.Value):
    def value(self, a):
        raise NotImplementedError

    def confidence(self, a):
        raise NotImplementedError

    def update_value(self, a):
        raise NotImplementedError

    def update_target(self, a):
        raise NotImplementedError

    # TODO optim_value
    # TODO optim_actions
    # TODO optim_action


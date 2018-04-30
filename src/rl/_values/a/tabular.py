from .value import A
import rl.values as rlvalues


# TODO move something to rlmisc
class Tabular(A, rlvalues.Tabular):
    def __init__(self, env):
        A.__init__(self, env)
        rlvalues.Tabular.__init__(self, (env.aspace,))

    def value(self, a):
        return rlvalues.Tabular.value(self, (a,))

    def update_value(self, a, value):
        rlvalues.Tabular.update_value(self, (a,), value)

    def update_target(self, a, value):
        rlvalues.Tabular.update_target(self, (a,), value)

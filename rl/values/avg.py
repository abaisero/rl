from .value import Value


class AvgValue(Value):
    def __init__(self, values):
        super().__init__(values[0].env)
        self.values = values

    def value(self, *args, **kwargs):
        return sum(val(*args, **kwargs) for val in self.values) / len(self.values)

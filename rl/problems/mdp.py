

# class ModelException(Exception):
#     pass


# class MABModel(object):
#     def sample_r(self, a):
#         raise NotImplementedError


# class CBModel(object):
#     def sample_s(self):
#         raise NotImplementedError

#     def sample_r(self, s, a):
#         raise NotImplementedError


# class MDPModel(object):
#     def sample_s0(self):
#         raise NotImplementedError

#     def sample_s1(self, s0, a):
#         raise NotImplementedError

#     def sample_r(self, s0, a, s1):
#         raise NotImplementedError

#     def sample_rs1(self, s0, a):
#         s1 = self.sample_s1(s0, a)
#         r = self.sample_r(s0, a, s1)
#         return r, s1


from . import Model, RLProblem


class MDPModel(Model):
    pass
    # def __init__(self, model):
    #     super(MDPModel, self).__init__(model)


class MDP(RLProblem):
    pass
    # def __init__(self, model):
    #     super(MDP, self).__init__(model)

from rl.values import Values_V2Q


# TODO implement Q flavor of value_iteration
# def Q(Q, sys):
#     delta = 1
#     while delta > 1e-8:
#         delta = 0
#         for s in sys.states():
#             for a in sys.actions(s):
#                 q = bellman.equation_optim.Q(Q, s, a, sys)
#                 delta = max(delta, abs(q - Q(s, a)))
#                 Q.update_value(s, a, q)


def V(V, sys, model):
    delta = 1
    while delta > 1e-8:
        delta = 0
        for s in sys.states():
            v = V.optim_value(s, sys.actions(s), model)
            delta = max(delta, abs(v - V(s)))
            V.update_value(s, v)

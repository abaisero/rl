# class dotpomdp(object):
#     def __init__(self, pomdp_name):
#         # TODO how to access pomdp folder.... and exception!
#         self.f = open(f'{pomdp_name}.pomdp')

#     def __enter__(self):
#         return self.f

#     def __enter__(self, type, value, traceback):
#         self.f.close()


from pkg_resources import resource_filename
import rl

from contextlib import contextmanager

@contextmanager
def dotpomdp(name):
    fname = resource_filename('rl', f'data/pomdps/{name}.pomdp')
    with open(fname) as f:
        yield f

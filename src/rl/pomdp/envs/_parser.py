from contextlib import contextmanager
from pkg_resources import resource_filename

from rl_parsers.pomdp import parse


@contextmanager
def _open(fname):
    try:
        f = open(fname)
    except FileNotFoundError:
        fname = resource_filename('rl', f'data/pomdp/{fname}')
        f = open(fname)

    yield f
    f.close()


import pytk.factory as factory
import rl.pomdp as pomdp
import numpy as np


def env(fname):
    with _open(fname) as f:
        dotpomdp = parse(f.read())

    if dotpomdp.values == 'cost':
        raise ValueError('I do not know how to handle `cost` values.')

    # TODO I think this should not be mean but something else..
    if np.any(dotpomdp.R.mean(axis=-1, keepdims=True) != dotpomdp.R):
        raise ValueError('I cannot handle rewards which depend on observations.')

    sfactory = factory.FactoryValues(dotpomdp.states)
    afactory = factory.FactoryValues(dotpomdp.actions)
    ofactory = factory.FactoryValues(dotpomdp.observations)

    env = pomdp.Environment(sfactory, afactory, ofactory)
    env.gamma = dotpomdp.discount

    if dotpomdp.start is None:
        start = np.ones(env.nstates) / env.nstates
    else:
        start = dotpomdp.start
    T = np.swapaxes(dotpomdp.T, 0, 1)
    O = np.stack([dotpomdp.O] * env.nstates)
    R = np.einsum('jik', dotpomdp.R.mean(axis=-1))

    s0model = pomdp.State0Distribution(env)
    s1model = pomdp.State1Distribution(env)
    omodel = pomdp.ObsDistribution(env)
    rmodel = pomdp.RewardDistribution(env)

    s0model.array = start
    s1model.array = T
    omodel.array = O
    rmodel.array = R

    env.model = pomdp.Model(env, s0model, s1model, omodel, rmodel)
    return env

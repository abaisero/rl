from contextlib import contextmanager
from pkg_resources import resource_filename

from rl_parsers.pomdp import parse as parse_pomdp

import indextools
import rl.pomdp as pomdp
import numpy as np


@contextmanager
def _open(fname):
    try:
        f = open(fname)
    except FileNotFoundError:
        fname = resource_filename('rl', f'data/pomdp/{fname}')
        f = open(fname)

    yield f
    f.close()


def from_fname(fname):
    with _open(fname) as f:
        dotpomdp = parse_pomdp(f.read())

    if dotpomdp.values == 'cost':
        raise ValueError('I do not know how to handle `cost` values.')

    # TODO I think this should not be mean but something else..
    if np.any(dotpomdp.R.mean(axis=-1, keepdims=True) != dotpomdp.R):
        raise ValueError('I cannot handle rewards which depend on observations.')

    sspace = indextools.DomainSpace(dotpomdp.states)
    aspace = indextools.DomainSpace(dotpomdp.actions)
    ospace = indextools.DomainSpace(dotpomdp.observations)
    gamma = dotpomdp.discount
    domain = pomdp.Domain(sspace, aspace, ospace, gamma=gamma)

    if dotpomdp.start is None:
        start = np.ones(sspace.nelems) / sspace.nelems
    else:
        start = dotpomdp.start
    T = np.swapaxes(dotpomdp.T, 0, 1)
    O = np.stack([dotpomdp.O] * sspace.nelems)
    R = np.einsum('jik', dotpomdp.R.mean(axis=-1))

    s0model = pomdp.State0Distribution(domain)
    s1model = pomdp.State1Distribution(domain)
    omodel = pomdp.ObsDistribution(domain)
    rmodel = pomdp.RewardDistribution(domain)

    s0model.array = start
    s1model.array = T
    omodel.array = O
    rmodel.array = R

    domain.model = pomdp.Model(s0model, s1model, omodel, rmodel)
    return domain

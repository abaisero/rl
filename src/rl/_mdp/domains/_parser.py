from contextlib import contextmanager
from pkg_resources import resource_filename

from rl_parsers.mdp import parse as parse_mdp

import indextools
import rl.mdp as mdp
import numpy as np


@contextmanager
def _open(fname):
    try:
        f = open(fname)
    except FileNotFoundError:
        fname = resource_filename('rl', f'data/mdp/{fname}')
        f = open(fname)

    yield f
    f.close()


def from_fname(fname):
    with _open(fname) as f:
        dotmdp = parse_mdp(f.read())

    if dotmdp.values == 'cost':
        raise ValueError('I do not know how to handle `cost` values.')

    # TODO I think this should not be mean but something else..
    if np.any(dotmdp.R.mean(axis=-1, keepdims=True) != dotmdp.R):
        raise ValueError('I cannot handle rewards which depend on observations.')

    sspace = indextools.DomainSpace(dotmdp.states)
    aspace = indextools.DomainSpace(dotmdp.actions)
    gamma = dotmdp.discount
    domain = mdp.Domain(sspace, aspace, gamma=gamma)

    if dotmdp.start is None:
        start = np.ones(sspace.nelems) / sspace.nelems
    else:
        start = dotmdp.start
    T = np.swapaxes(dotmdp.T, 0, 1)
    R = np.einsum('jik', dotmdp.R)

    s0model = mdp.State0Distribution(domain)
    s1model = mdp.State1Distribution(domain)
    rmodel = mdp.RewardDistribution(domain)

    s0model.array = start
    s1model.array = T
    rmodel.array = R

    domain.model = mdp.Model(s0model, s1model, rmodel)
    return domain

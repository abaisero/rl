def run_episode(sys, model, policy, s, verbose=False):
    if verbose:
        print '---'

    e = []
    while not s.terminal:
        actions = sys.actions(s)
        a = policy.sample(s, actions)
        r, s1 = model.sample_rs1(s, a)
        e.append((s, a, r, s1))
        if verbose:
            print '{}, {}, {}, {}'.format(s, a, r, s1)
        s = s1
    return e

def make_returns(episode, gamma, first_visit=True):
    returns = dict()

    R = 0.
    for s, a, r, s1 in episode[::-1]:
        R = r + gamma * R
        if first_visit:
            returns[s, a] = R
        else:
            returns.setdefault((s, a), []).append(R)

    return returns

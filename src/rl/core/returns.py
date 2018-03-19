def discounted_sum(sys, episode):
    """J \eqdot \sum_{t=1}^T r_t \lambda^t"""
    G = 0.
    for _, _, feedback in episode:
        G = sys.env.gamma * G + feedback.r
    return G


def longterm_average(sys, episode):
    r"""\nu \eqdot \frac{1}{T} \sum_{t=1}^T r_t"""
    return sum(feedback.r for _, _, feedback in episode) / len(episode)

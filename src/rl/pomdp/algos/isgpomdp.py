import logging
logger = logging.getLogger(__name__)


def isgpomdp(params, policy, env, nsteps=100, beta=None):
    """ "Scaling Internal-State Policy-Gradient Methods for POMDPs"
        - D. Aberdeen, J. Baxter """

    if beta is None:
        beta = env.gamma

    g = 0.
    z, d = 0, 0

    econtext = env.new_context()
    pcontext = policy.new_context(params)
    while econtext.t < nsteps:
        a = policy.sample_a(params, pcontext)
        feedback, _ = env.step(econtext, a)
        # NOTE econtext already takes step here
        pcontext1 = policy.step(params, pcontext, feedback, inline=False)

        g += (feedback.r - g) / econtext.t
        z = beta * z + policy.dlogprobs(
            params, pcontext, a, feedback, pcontext1)
        d += (feedback.r * z - d) / econtext.t
        pcontext = pcontext1

    return g, d

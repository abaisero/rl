import types
import copy


# def isgpomdp(params, policy, env, nsteps=100, beta=None):
#     """ "Scaling Internal-State Policy-Gradient Methods for POMDPs"
#         - D. Aberdeen, J. Baxter """

#     if beta is None:
#         beta = env.gamma

#     g = 0.
#     z, d = 0, 0

#     econtext = env.new_context()
#     pcontext = policy.new_context(params)
#     while econtext.t < nsteps:
#         a = policy.sample_a(params, pcontext)
#         feedback, _ = env.step(econtext, a)
#         # NOTE econtext already takes step here
#         pcontext1 = policy.step(params, pcontext, feedback, inline=False)

#         g += (feedback.r - g) / econtext.t
#         z = beta * z + policy.dlogprobs(
#             params, pcontext, a, feedback, pcontext1)
#         d += (feedback.r * z - d) / econtext.t
#         pcontext = pcontext1

#     return g, d


class IsGPOMDP:
    """ "Scaling Internal-State Policy-Gradient Methods for POMDPs"
        - D. Aberdeen, J. Baxter """
    type_ = 'episodic'

    def __init__(self, policy, beta):
        self.policy = policy
        self.beta = beta

    def new_context(self):
        return types.SimpleNamespace(
            obj=0.,  # objective value
            elig=0.,  # eligibility trace
            grad=0.,  # gradient
        )

    def step(self, params, acontext, econtext, pcontext, a, feedback,
             pcontext1, *, inline=False):
        """ "Scaling Internal-State Policy-Gradient Methods for POMDPs"
            - D. Aberdeen, J. Baxter """
        if not inline:
            acontext = copy.copy(acontext)

        dlogprobs = self.policy.dlogprobs(params, pcontext, a, feedback,
                                          pcontext1)

        acontext.obj += (feedback.r - acontext.obj) / (econtext.t + 1)
        acontext.elig = self.beta * acontext.elig + dlogprobs
        acontext.grad += ((feedback.r * acontext.elig - acontext.grad) /
                          (econtext.t + 1))

        return acontext

    def episode(self, params, policy, env, nsteps):
        econtext = env.new_context()
        pcontext = policy.new_context(params)
        acontext = self.new_context()
        while econtext.t < nsteps:
            a = policy.sample_a(params, pcontext)
            feedback, econtext1 = env.step(econtext, a)
            pcontext1 = policy.step(params, pcontext, feedback)
            pscore = policy.dlogprobs(params, pcontext, a, feedback, pcontext1)

            self.step(params, acontext, econtext, feedback, pscore,
                      inline=True)

            econtext = econtext1
            pcontext = pcontext1

        return acontext

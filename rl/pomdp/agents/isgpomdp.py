import logging
logger = logging.getLogger(__name__)

from .agent import Agent


class IsGPOMDP(Agent):
    logger = logging.getLogger(f'{__name__}.IsGPOMDP')

    def reset(self):
        self.policy.reset()
        self.restart()

    def restart(self):
        self.policy.restart()
        self.ihistory = []

    def feedback(self, sys, context, a, feedback):
        self.logger.debug(f'feedback() \t; {context} \t; a={a} \t; {feedback}')

        o = feedback.o

        icontext = self.policy.context
        ifeedback = self.policy.feedback(o)
        self.ihistory.append((icontext, ifeedback))

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        G = 0.

        history = zip(reversed(episode), reversed(self.ihistory))
        for (context, a, feedback), (icontext, ifeedback) in history:
            t = context.t
            o, r = feedback.o, feedback.r
            n = icontext.n
            n1 = ifeedback.n1

            G = r + self.env.gamma * G

            # TODO let some other method take care of how to do the gradient descent (normalized or not, etc...)
            alpha = .001

            daparams = self.policy.amodel.dlogprobs(n, a)
            # daparams /= np.sqrt((daparams * daparams).sum())
            self.policy.amodel.params += alpha * G * daparams
            # self.policy.amodel.params += alpha * (self.env.gamma ** t ) * G * daparams

            doparams = self.policy.omodel.dlogprobs(n, o, n1)
            # doparams /= np.sqrt((doparams * doparams).sum())
            self.policy.omodel.params += alpha * G * doparams
            # self.policy.omodel.params += alpha * (self.env.gamma ** t) * G * doparams

import logging
logger = logging.getLogger(__name__)

from .agent import Agent

import numpy as np
import numpy.random as rnd


# TODO make both first visit (done?) and every visit

class MonteCarloES(Agent):
    logger = logging.getLogger(f'{__name__}.MonteCarloES')

    def __init__(self, name, env, policy, Q):
        super().__init__(name, env, policy)
        self.Q = Q
        self.Glists = dict()

    def act(self, context):
        # Random first action
        if context.t == 0:
            return rnd.choice(list(self.env.actions))
        return super().act(context)

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        G, Gs = 0., dict()
        for context, a, feedback in reversed(episode):
            s = context.s
            r = feedback.r
            G = r + self.env.gamma * G
            Gs[s, a] = G

        for (s, a), G in Gs.items():
            Glist = self.Glists.setdefault((s, a), [])
            Glist.append(G)
            self.Q[s, a] = np.mean(Glist)
            logger.debug(f'agent={self.name} \t; s={s} \t; a={a} \t; Q={self.Q(s, a)}')


class MonteCarloControl(Agent):
    logger = logging.getLogger(f'{__name__}.MonteCarloControl')

    def __init__(self, name, env, policy, Q):
        super().__init__(name, env, policy)
        self.Q = Q
        self.Glists = dict()

    def feedback_episode(self, sys, episode):
        self.logger.debug(f'feedback_episode() \t; len(episode)={len(episode)}')

        G, Gs = 0., dict()
        for context, a, feedback in reversed(episode):
            s = context.s
            s1, r = feedback.s1, feedback.r
            gamma = .9
            G = r + gamma * G
            # G += r
            Gs[s, a] = G

        logger.debug(f'agent={self.name} \t; feedback_episode()')
        for (s, a), G in Gs.items():
            Glist = self.Glists.setdefault((s, a), [])
            Glist.append(G)
            self.Q[s, a] = np.mean(Glist)

            logger.debug(f'agent={self.name} \t; {s} \t; {a} \t; {self.Q(s, a)}')

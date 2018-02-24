from .horizon import Horizon

import itertools as itt

import logging
logger = logging.getLogger(__name__)


# TODO rename engine?
class System:
    """ A system keeps track of system state and running conditions """
    def __init__(self, env, model, horizon=None):
        if horizon is None:
            horizon = Horizon()

        self.env = env
        self.model = model
        self.horizon = horizon

        self.restart()

    def restart(self):
        self.s, = self.model.s0.sample()
        self.terminal = False

    @property
    def done(self):
        return self.s is None or self.terminal or self.horizon(self.t)

    @property
    def context(self):
        raise NotImplementedError

    def step(self, a):
        raise NotImplementedError

    def run(self, agent, nepisodes=1, *, callbacks=None, feedbacks=None, feedbacks_episode=None):
        if callbacks is None:
            callbacks = tuple()

        if feedbacks is None:
            feedbacks = tuple()

        if feedbacks_episode is None:
            feedbacks_episode = tuple()

        # TODO possibly specify new horizon object here...
        for ne in range(nepisodes):
            logger.info(f'Episode {ne} begins')

            self.restart()
            agent.restart()

            episode = []

            self.t = 0
            while not self.done:
                context = self.context
                a = agent.act(context)
                feedback = self.step(a)

                logger.info(f'e={ne} \t; t={self.t} \t; s={self.s} \t; {context} \t; a={a} \t; {feedback}')

                episode.append((context, a, feedback))

                self.t += 1

                agent.feedback(self, context, a, feedback)
                for cb in callbacks:
                    cb.feedback(self, context, a, feedback)
                for fb in feedbacks:
                    fb(self, context, a,feedback)

            agent.feedback_episode(self, episode)
            for cb in callbacks:
                cb.feedback_episode(self, episode)
            for fbe in feedbacks_episode:
                fbe(self, episode)

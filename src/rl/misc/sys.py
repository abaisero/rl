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
            context1 = self.context
            while not self.done:
                context = context1

                a = agent.act(context)
                feedback = self.step(a)

                logger.info(f'e={ne} \t; t={self.t} \t; s={self.s} \t; {context} \t; a={a} \t; {feedback}')

                self.t += 1
                context1 = self.context
                episode.append((context, a, feedback, context1))

                self.t += 1

                # TODO remove all this crap!
                agent.feedback(context, a, feedback, context1)
                for cb in callbacks:
                    cb.feedback(context, a, feedback, context1)
                for fb in feedbacks:
                    fb(context, a,feedback, context1)

            agent.feedback_episode(episode)
            for cb in callbacks:
                cb.feedback_episode(episode)
            for fbe in feedbacks_episode:
                fbe(episode)

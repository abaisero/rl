from .callback import Callback

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
# sns.set_style('darkgrid')


def qplot(q, shape, pdict, plt_init=None, delay=1/30):
    ni, nj = shape
    data = np.full(shape, np.nan)

    percentiles = list(pdict)

    ax = None
    i, y = 0, q.get()
    data.itemset(i, y)

    import time
    from queue import Empty
    starttime = time.time()
    while True:
        while True:
            try:
                y = q.get_nowait()
                i += 1
            except Empty:
                break
            else:
                if y is None: break
                data.itemset(i, y)

        if y is None: break

        if ax is not None:
            ax.clear()
        ax = sns.tsplot(data=data, estimator=np.nanmean, **pdict)

        time.sleep(delay - ((time.time() - starttime) % delay))


class SNS_PercentilePlotter(Callback):
    def __init__(self, env, shape, pdict, *, plt_init=None):
        super().__init__(env)

        self.q = mp.Queue()
        self.p = mp.Process(target=qplot, args=(self.q, shape, pdict, plt_init))
        self.p.daemon = True
        self.p.start()

    def feedback_episode(self, sys, episode):
        G = 0.
        for context, a, feedback in reversed(episode):
            r = feedback.r
            G = r + self.env.gamma * G

        self.q.put(G)

    def close(self):
        self.q.put(None)
        self.p.join()

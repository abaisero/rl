from .callback import Callback

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def qplot(q, shape, pdict, plt_init=None, delay=1/30):
#     ni, nj = shape
#     data = np.full(shape, np.nan)
#     # df = pd.DataFrame(data)
#     # xdata = list(range(nj))

#     percentiles = list(pdict)

#     ldict = dict()
#     i, y = 0, q.get()
#     data.itemset(i, y)

#     import time
#     from queue import Empty
#     starttime = time.time()
#     while True:
#         while True:
#             try:
#                 y = q.get_nowait()
#                 i += 1
#             except Empty:
#                 break
#             else:
#                 if y is None: break
#                 data.itemset(i, y)

#         if y is None: break

#         if len(ldict) == 0:
#             ax = plt.subplot()

#             ppdata = zip(percentiles, np.nanpercentile(data, percentiles, axis=0))
#             for p, pdata in ppdata:
#                 ldict[p], = ax.plot(pdata, **pdict[p])

#             if plt_init is not None:
#                 plt_init()
#         else:
#             ppdata = zip(percentiles, np.nanpercentile(data, percentiles, axis=0))
#             # for p, pdata in zip(percentiles, np.nanpercentile(data, percentiles, axis=0)):
#             for p, pdata in ppdata:
#                 ldict[p].set_ydata(pdata)

#             ax.relim()
#             ax.autoscale_view()
#             plt.draw()
#             plt.pause(.000000000001)

#         if y is None: break

#         time.sleep(delay - ((time.time() - starttime) % delay))


class PLT_PercentilePlotter(Callback):
    def __init__(self, env, shape, pdict, *, plt_init=None):
        super().__init__(env)
        self.data = np.full(shape, np.nan)
        self.idx = 0

        self.q = mp.Queue()
        self.p = mp.Process(target=self.target, args=(self.q, shape, pdict, plt_init))
        self.p.daemon = True
        self.p.start()


    @staticmethod
    def target(q, shape, pdict, plt_init=None, delay=1/30):
        data = np.full(shape, np.nan)
        percentiles = list(pdict)
        ldict = dict()

        d = q.get()
        data.itemset(*d)

        import time
        from queue import Empty
        starttime = time.time()
        while True:
            while True:
                try:
                    d = q.get_nowait()
                except Empty:
                    break
                else:
                    if d is None: break
                    data.itemset(*d)

            if d is None: break

            nanpercentiles = np.nanpercentile(data, percentiles, axis=0)
            try:
                for p, pdata in zip(percentiles, nanpercentiles):
                    ldict[p].set_ydata(pdata)
            except KeyError:
                ax = plt.subplot()
                for p, pdata in zip(percentiles, nanpercentiles):
                    ldict[p], = ax.plot(pdata, **pdict[p])
                try:
                    plt_init()
                except TypeError:
                    pass
            else:
                ax.relim()
                ax.autoscale_view()
                plt.draw()

            plt.pause(delay - ((time.time() - starttime) % delay))

    def feedback_episode(self, sys, episode):
        G = 0.
        for context, a, feedback in episode:
            r = feedback.r
            G = r + self.env.gamma * G

        self.q.put((self.idx, G))
        self.idx += 1

    def close(self):
        self.q.put(None)
        self.p.join()

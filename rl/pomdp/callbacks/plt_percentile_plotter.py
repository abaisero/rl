from .callback import Callback

import multiprocessing as mp
import queue

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

    def feedback_episode(self, sys, episode):
        # TODO it would be nice to handle this differently...
        # what if there are multiple callbacks which all want to process the episode return???

        G = 0.
        for context, a, feedback in episode:
            r = feedback.r
            G = r + self.env.gamma * G

        self.q.put((self.idx, G))
        self.idx += 1

    def close(self):
        # TODO currently unused
        self.q.put(None)
        self.p.join()

    @staticmethod
    def target(q, shape, pdict, plt_init=None):
        #  initialize data and plot variables
        data = np.full(shape, np.nan)
        percentiles = list(pdict)
        ldict = dict()

        #  initialize plot
        ax = plt.subplot()

        #  wait until new consumable becomes available
        for idx, d in iter(q.get, None):
            data.itemset(idx, d)

            #  consume as much as possible before plotting
            try:
                for idx, d in iter(q.get_nowait, None):
                    data.itemset(idx, d)
            except queue.Empty:
                pass

            #  (update) plot
            nanpercentiles = np.nanpercentile(data, percentiles, axis=0)
            try:
                for p, pdata in zip(percentiles, nanpercentiles):
                    ldict[p].set_ydata(pdata)
            except KeyError:
                for p, pdata in zip(percentiles, nanpercentiles):
                    ldict[p], = ax.plot(pdata, **pdict[p])

                try:
                    plt_init()
                except TypeError:
                    pass
            else:
                ax.relim()
                ax.autoscale_view()
                # plt.draw()  #  redundant by plt.pause
                plt.pause(1e-10)

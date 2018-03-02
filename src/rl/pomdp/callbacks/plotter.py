from .callback import Callback


class Plotter:
    def __init__(self, env, shape, pdict, *, plt_init=None):
        super().__init__(env)
        self.data = np.full(shape, np.nan)
        self.idx = 0

        self.q = mp.Queue()
        self.p = mp.Process(target=self.target, args=(self.q, shape, pdict, plt_init))
        self.p.daemon = True
        self.p.start()

    # TODO separate callback to compute this stuff...
    def feedback_episode(self, sys, episode):
        # TODO it would be nice to handle this differently...
        # what if there are multiple callbacks which all want to process the episode return???

        # TODO isn't this opposite?! no.. no.. ok
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
        ax_pp = plt.subplot(211)
        ax_rp = plt.subplot(212, sharex=ax_pp, sharey=ax_pp)

        #  initialize data and plot variables
        data = np.full(shape, np.nan)
        percentiles = list(pdict)
        pp_ldict = dict()
        rp_lines = ax_rp.plot(data.T, linewidth=1)

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
            # TODO maybe use list (like rp) instead of ldist
            try:
                for p, pdata in zip(percentiles, nanpercentiles):
                    pp_ldict[p].set_ydata(pdata)
            except KeyError:
                for p, pdata in zip(percentiles, nanpercentiles):
                    pp_ldict[p], = ax_pp.plot(pdata, **pdict[p])

                try:
                    plt.sca(ax_pp)
                    plt_init()
                except TypeError:
                    pass
            else:
                ax_pp.relim()
                ax_pp.autoscale_view()

            #  (update) radar plot
            anynan = np.isnan(data).any(axis=1)
            for l, d, an in zip(rp_lines, data, anynan):
                if an:
                    l.set_ydata(d)
                elif l.axes is not None:
                    l.remove()

            # plt.draw()  #  redundant by plt.pause
            plt.pause(1e-10)

import pyqtgraph as pg

import numpy as np


class PPlotWidget(pg.GraphicsLayoutWidget):
    def setup(self, data, pdict, **kwargs):
        self.data = data
        self.pdict = pdict
        self.percentiles = list(pdict)

        self.addLabel(**kwargs.get('window', {}))
        self.nextRow()

        plt_pp = self.addPlot(
            title='Run Percentiles',
            left=kwargs.get('labels', {}).get('left', (None,)),
        )
        self.nextRow()

        plt_pp.showAxis('right')
        plt_pp.addLegend(offset=(-30, -30))
        plt_pp.showGrid(x=True, y=True)

        self.curves_pp = {p: plt_pp.plot(**pdata)
                          for p, pdata in pdict.items()}

        # TODO change to pg colormap
        import matplotlib.cm as cm
        cmap = cm.tab10_r
        plt_rp = self.addPlot(
            title='Individual Runs',
            left=kwargs.get('labels', {}).get('left', (None,)),
            bottom=kwargs.get('labels', {}).get('bottom', (None,)),
        )

        plt_rp.showAxis('right')
        plt_rp.showGrid(x=True, y=True)
        self.curves_rp = [plt_rp.plot(pen=cmap(i % cmap.N, bytes=True))
                          for i in range(data.shape[0])]
        # TODO too many curves will break this

        # def setRange(rect=None, xRange=None, yRange=None, *args, **kwds):
        #     if not kwds.get('disableAutoRange', True):
        #         if yRange is not None:
        #             yRange[0] = 0
        #     # pg.ViewBox.setRange(vb, rect, xRange, yRange, *args, **kwds)
        # plt_rp.vb.setRange = setRange

        plt_rp.setYLink(plt_pp)
        plt_rp.setXLink(plt_pp)

        try:
            plt_rp.setRange(
                xRange=kwargs.get('ranges', {}).get('x'),
                yRange=kwargs.get('ranges', {}).get('y'),
            )
        except Exception:
            pass

        return self

    def update(self):
        nanpercentiles = np.nanpercentile(self.data, self.percentiles, axis=0)
        isnan = np.isnan(self.data)
        anynan = isnan.any(axis=1)
        allnan = isnan.all(axis=1)

        for p, pdata in zip(self.percentiles, nanpercentiles):
            self.curves_pp[p].setData(pdata)

        # TODO optimize this.. don't use visible/non-visible stuff..
        # actually add/remove plots!!!
        for c, d, anyn, alln in zip(self.curves_rp, self.data, anynan, allnan):
            if anyn and not alln:
                c.setVisible(True)
                c.setData(d)
            else:
                c.setVisible(False)

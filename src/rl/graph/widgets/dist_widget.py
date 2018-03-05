import pyqtgraph as pg

import numpy as np


class DistWidget(pg.GraphicsLayoutWidget):
    def setup(self, data, ylabels, xlabels):
        self.setData(data)

        ny = len(ylabels)
        nx = len(xlabels)

        self.plots = np.empty((ny, nx), dtype=object)
        self.curves = np.empty((ny, nx), dtype=object)

        self.nextCol()
        for xlab in xlabels:
            self.addLabel(text=xlab, bold=True)

        for yi, ylab in enumerate(ylabels):
            self.nextRow()
            self.addLabel(text=ylab, bold=True, angle=-90)

            for xi in range(nx):
                plot = self.addPlot()
                plot.enableAutoRange(False)
                plot.setYRange(0, 1)
                plot.setMouseEnabled(x=False, y=False)
                plot.hideButtons()
                curve = plot.plot()

                self.plots[yi, xi] = plot
                self.curves[yi, xi] = curve

        return self

    def setData(self, data):
        self.data = data

    def update(self):
        for (yi, xi), curve in np.ndenumerate(self.curves):
            curve.setData(self.data[:, yi, xi])

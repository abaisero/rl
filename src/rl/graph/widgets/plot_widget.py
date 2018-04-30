import pyqtgraph as pg

import numpy as np


class PlotWidget(pg.GraphicsLayoutWidget):
    def setup(self, data, **kwargs):
        self.data = data

        self.addLabel(**kwargs.get('window', {}))
        self.nextRow()

        plot = self.addPlot(
            title='Plot',
            left=kwargs.get('labels', {}).get('left', (None,)),
        )
        self.nextRow()

        plot.showAxis('right')
        plot.addLegend(offset=(-30, -30))
        plot.showGrid(x=True, y=True)

        plot.setMouseEnabled(x=True, y=False)
        # plot.setLimits(xMin=0, xMax=data.shape[-1])
        # plot.setLimits(minXRange=100, maxXRange=data.shape[-1])

        for line in kwargs.get('lines', []):
            plot.addItem(pg.InfiniteLine(**line))
        # TODO multiple...?

        self.cshape = data.shape[:-1]
        self.curve = np.empty(self.cshape, dtype=object)
        for idx in np.ndindex(self.cshape):
            self.curve[idx] = plot.plot()

        try:
            plot.setRange(
                xRange=kwargs.get('ranges', {}).get('x'),
                yRange=kwargs.get('ranges', {}).get('y'),
            )
        except Exception:
            pass

        return self

    def update(self):
        for idx in np.ndindex(self.cshape):
            self.curve[idx].setData(self.data[idx])
        # print(self.data.shape)
        # self.curve.setData(self.data.sum(axis=1))

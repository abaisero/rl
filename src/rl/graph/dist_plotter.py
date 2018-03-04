import multiprocessing as mp
import threading
import queue

import numpy as np


def thread_target(l, q, data):
    for idx, value in iter(q.get, None):
        with l:
            data[idx] = value.T


# TODO fsc_plotter also has a plotter like this... reuse that!!!  only create window and widget here!!
def process_target(shape, q, **kwargs):
    neps, nx, ny = shape
    shape = neps, ny, nx
    data = np.full(shape, np.nan)

    l = mp.Lock()

    t = threading.Thread(target=thread_target, args=(l, q, data))
    t.deamon = True
    t.start()

    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg

    app = QtGui.QApplication([])
    print(app, type(app))
    # TODO do this once?...

    # TODO some of these need a selector
    layout = pg.GraphicsLayoutWidget()
    layout.show()

    xlabels = kwargs.get('xlabels', list(range(nx)))
    ylabels = kwargs.get('ylabels', list(range(ny)))

    layout.nextCol()
    for xlab in xlabels:
        layout.addLabel(text=xlab, bold=True)

    plots = np.empty((ny, nx), dtype=object)
    curves = np.empty((ny, nx), dtype=object)
    for yi, ylab in enumerate(ylabels):
        layout.nextRow()
        layout.addLabel(text=ylab, bold=True, angle=-90)

        for xi in range(nx):
            plot = layout.addPlot()
            plot.setYRange(0, 1)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            curve = plot.plot()

            plots[yi, xi] = plot
            curves[yi, xi] = curve

    def update():
        for yi in range(ny):
            for xi in range(nx):
                with l:
                    d = data[:, yi, xi]
                curves[yi, xi].setData(d)

        # app.processEvents()  ## force complete redraw for every plot

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000 / 10)

    # QtGui.QApplication.instance().exec_()
    import sys
    sys.exit(app.exec_())
    # QtGui.QApplication.instance().exec_()


def distplot(shape, **kwargs):
    q = mp.Queue()
    p = mp.Process(target=process_target, args=(shape, q), kwargs=kwargs)
    p.daemon = True
    p.start()
    return q, p

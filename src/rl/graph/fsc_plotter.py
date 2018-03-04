import multiprocessing as mp
import threading
import queue

import numpy as np

import rl.graph.design

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from .distplot_stackful import DistPlot_Stackful
from .fsc_window import FSC_Window


# TODO use QThread...
# TODO quit when done sending...
class MyThread(QtCore.QThread):
    def __init__(self, l, q, adata, odata):
        super().__init__()
        self.l = l
        self.q = q
        self.adata = adata
        self.odata = odata

    def run(self):
        for idx, adist, odist in iter(self.q.get, None):
            self.adata[idx] = adist.T
            self.odata[idx] = odist.T


# def thread_target(l, q, adata, odata):
#     for idx, adist, odist in iter(q.get, None):
#         adata[idx] = adist.T
#         odata[idx] = odist.T


class dist_gw:
    def __init__(self, ylabels, xlabels, data, idx):
        self.ylabels = ylabels
        self.xlabels = xlabels
        self.data = data
        self.idx = idx

        ny = len(ylabels)
        nx = len(xlabels)
        self.ny = ny
        self.nx = nx

        plots = np.empty((ny, nx), dtype=object)
        curves = np.empty((ny, nx), dtype=object)
        self.curves = curves

        gw = pg.GraphicsLayoutWidget()
        gw.nextCol()
        for xlab in xlabels:
            gw.addLabel(text=xlab, bold=True)

        for yi, ylab in enumerate(ylabels):
            gw.nextRow()
            gw.addLabel(text=ylab, bold=True, angle=-90)

            for xi in range(nx):
                plot = gw.addPlot()
                plot.enableAutoRange(False)
                plot.setYRange(0, 1)
                plot.setMouseEnabled(x=False, y=False)
                plot.hideButtons()
                curve = plot.plot()

                plots[yi, xi] = plot
                curves[yi, xi] = curve

        self.gw = gw

    def update(self):
        for yi in range(self.ny):
            for xi in range(self.nx):
                # self.curves[yi, xi].setData(self.data[:, yi, xi])
                # TODO I probably want the slice to be the last index... to make sure no copies need to be made....
                self.curves[yi, xi].setData(self.data[self.idx(yi, xi)])

                # with l:
                #     d = data[yi, xi]
                # curves[yi, xi].setData(d)



def process_target(q, nepisodes, alabels, nlabels, olabels):
    na = len(alabels)
    nn = len(nlabels)
    no = len(olabels)

    ashape = nepisodes, na, nn
    adata = np.full(ashape, np.nan)
    oshape = nepisodes, nn, no, nn
    odata = np.full(oshape, np.nan)

    l = mp.Lock()

    app = QtGui.QApplication([])

    agw = dist_gw(alabels, nlabels, adata, lambda yi, xi: (slice(None), yi, xi))
    # ogws = [dist_gw(nlabels, nlabels, odata[:, :, oi, :]) for oi in range(no)]
    # ngws = [dist_gw(nlabels, olabels, odata[:, :, :, ni]) for ni in range(no)]
    def oidx(oi): return lambda yi, xi: (slice(None), yi, oi, xi)
    ogws = [dist_gw(nlabels, nlabels, odata, oidx(oi)) for oi in range(no)]
    def nidx(ni): return lambda yi, xi: (slice(None), yi, xi, ni)
    ngws = [dist_gw(nlabels, olabels, odata, nidx(ni)) for ni in range(nn)]

    tab2 = DistPlot_Stackful()
    for olabel, ogw in zip(olabels, ogws):
        tab2.addWidget(ogw.gw, olabel)

    tab3 = DistPlot_Stackful()
    for nlabel, ngw in zip(nlabels, ngws):
        tab3.addWidget(ngw.gw, nlabel)

    win = FSC_Window()
    win.addTab(agw.gw, 'A-Strategy')
    win.addTab(tab2, 'O-Strategy (obs)')
    win.addTab(tab3, 'O-Strategy (node)')
    win.show()

    # TODO also stop the timer when thread dies!!!
    def update(updateAll=False):
        print('UPDATING')
        i = win.tabWidget.currentIndex()
        if i == 0 or updateAll:
            agw.update()
        elif i == 1 or updateAll:
            for ogw in ogws:
                ogw.update()
        elif i == 2 or updateAll:
            for ngw in ngws:
                ngw.update()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000)

    def endtimer():
        nonlocal timer
        timer.stop()
        update(updateAll=True)

    # timer.start(1000 / 10)
    # t = threading.Thread(target=thread_target, args=(l, q, adata, odata))
    # t.deamon = True
    # t.start()
    t = MyThread(l, q, adata, odata)
    t.finished.connect(endtimer)
    t.start()

    import sys
    sys.exit(app.exec_())


def fscplot(fsc, nepisodes):
    ashape = fsc.amodel.shape
    oshape = fsc.omodel.shape

    alabels = fsc.env.afactory.values
    nlabels = fsc.nfactory.values
    olabels = fsc.env.ofactory.values

    q = mp.Queue()
    p = mp.Process(target=process_target, args=(q, nepisodes, alabels, nlabels, olabels))
    p.daemon = True
    p.start()
    return q, p

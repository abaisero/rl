import multiprocessing as mp
import threading
import queue

import numpy as np

import rl.graph.design

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from .distplot_stackful import DistPlot_Stackful
from .fsc_window import FSC_Window


class DataThread(QtCore.QThread):
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


# class dist_gw(QtGui.QWidget):
class dist_gw(pg.GraphicsLayoutWidget):
    def __init__(self, ylabels, xlabels, data, *, parent=None, **kwargs):
        super().__init__(parent)

        self.__ylabels = ylabels
        self.__xlabels = xlabels
        self.__data = data

        ny = len(ylabels)
        nx = len(xlabels)
        self.__ny = ny
        self.__nx = nx

        self.__plots = np.empty((ny, nx), dtype=object)
        self.__curves = np.empty((ny, nx), dtype=object)

        # gw = pg.GraphicsLayoutWidget(self)
        # gw.nextCol()
        self.nextCol()
        for xlab in xlabels:
            # gw.addLabel(text=xlab, bold=True)
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

                self.__plots[yi, xi] = plot
                self.__curves[yi, xi] = curve

        # self.gw = gw

    def _update(self):
        for yi in range(self.__ny):
            for xi in range(self.__nx):
                self.__curves[yi, xi].setData(self.__data[:, yi, xi])



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

    agw = dist_gw(alabels, nlabels, adata)
    # TODO create one only... not a whole list!!!
    ogws = [dist_gw(nlabels, nlabels, odata[:, :, oi, :]) for oi in range(no)]
    ngws = [dist_gw(nlabels, olabels, odata[:, :, :, ni]) for ni in range(nn)]

    tab2 = DistPlot_Stackful()
    for olabel, ogw in zip(olabels, ogws):
        tab2.addWidget(ogw, olabel)

    tab3 = DistPlot_Stackful()
    for nlabel, ngw in zip(nlabels, ngws):
        tab3.addWidget(ngw, nlabel)

    gui = FSC_Window()
    gui.setWindowTitle('FSC')
    gui.addTab(agw, 'A-Strategy')
    gui.addTab(tab2, 'O-Strategy (obs)')
    gui.addTab(tab3, 'O-Strategy (node)')
    gui.show()

    def update(updateAll=False):
        i = gui.tabWidget.currentIndex()
        if i == 0 or updateAll:
            agw._update()
        if i == 1 or updateAll:
            for ogw in ogws:
                ogw._update()
        if i == 2 or updateAll:
            for ngw in ngws:
                ngw._update()

    update(updateAll=True)
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
    t = DataThread(l, q, adata, odata)
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


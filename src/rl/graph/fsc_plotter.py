import multiprocessing as mp
import threading
import queue

import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from .widgets import DistWidget, DistComboWidget, FSCWindow


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
    gui = FSCWindow()
    gui.setWindowTitle('FSC')

    dws = [
        DistWidget().setup(adata, alabels, nlabels),
        DistComboWidget().setup(odata, nlabels, nlabels, olabels, 2),
        DistComboWidget().setup(odata, nlabels, olabels, nlabels, 3),
    ]

    gui.addTab(dws[0], 'A-Strategy')
    gui.addTab(dws[1], 'O-Strategy (obs)')
    gui.addTab(dws[2], 'O-Strategy (node)')

    gui.show()

    def update(updateAll=False):
        if updateAll:
            for dw in dws:
                dw.update()
        else:
            i = gui.tabWidget.currentIndex()
            dws[i].update()

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


import multiprocessing as mp
import threading
import queue
import sys

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from .widgets import DistWidget, DistComboWidget, FSCWindow

import numpy as np


class DataThread(QtCore.QThread):
    def __init__(self, q, adata, odata):
        super().__init__()
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

    app = QtGui.QApplication([])
    gui = FSCWindow()
    gui.setWindowTitle('FSC')

    gui.addTab(
        DistWidget().setup(adata, alabels, nlabels),
        'A-Strategy',
    )

    gui.addTab(
        DistComboWidget().setup(odata, nlabels, nlabels, olabels, 2),
        'O-Strategy (obs)',
    )

    gui.addTab(
        DistComboWidget().setup(odata, nlabels, olabels, nlabels, 3),
        'O-Strategy (node)',
    )

    gui.update(updateAll=True)
    gui.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(gui.update)
    timer.start(1000)

    def endtimer():
        timer.stop()
        gui.update(updateAll=True)

    t = DataThread(q, adata, odata)
    t.finished.connect(endtimer)
    t.start()

    sys.exit(app.exec_())


def fscplot(fsc, nepisodes):
    alabels = fsc.env.afactory.values
    nlabels = fsc.nfactory.values
    olabels = fsc.env.ofactory.values

    q = mp.Queue()
    p = mp.Process(target=process_target, args=(q, nepisodes, alabels, nlabels, olabels))
    p.daemon = True
    p.start()
    return q, p

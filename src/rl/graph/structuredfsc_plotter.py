import multiprocessing as mp
import threading
import queue
import sys

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from .widgets import DistWidget, DistComboWidget, FSCWindow

import numpy as np


class DataThread(QtCore.QThread):
    def __init__(self, q, adata, ndata):
        super().__init__()
        self.q = q
        self.adata = adata
        self.ndata = ndata

    def run(self):
        for idx, adist, ndist in iter(self.q.get, None):
            self.adata[idx] = adist.T
            self.ndata[idx] = ndist.T


def process_target(q, nepisodes, alabels, nlabels, olabels, amask, nmask):
    na = len(alabels)
    nn = len(nlabels)
    no = len(olabels)

    ashape = nepisodes, na, nn
    adata = np.full(ashape, np.nan)
    nshape = nepisodes, nn, no, nn
    ndata = np.full(nshape, np.nan)
    nmask = np.stack([nmask] * no, axis=1)

    app = QtGui.QApplication([])
    gui = FSCWindow().setup()
    gui.setWindowTitle('FSC')

    gui.addTab(
        DistWidget().setup(adata, alabels, nlabels, amask),
        'A-Strategy',
    )

    gui.addTab(
        DistComboWidget().setup(ndata, nlabels, nlabels, olabels, 2, nmask),
        'O-Strategy | obs',
    )

    gui.addTab(
        DistComboWidget().setup(ndata, nlabels, olabels, nlabels, 3, nmask),
        'O-Strategy | node',
    )

    gui.update(updateAll=True)
    gui.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(gui.update)
    timer.start(1000)

    def endtimer():
        timer.stop()
        gui.update(updateAll=True)

    t = DataThread(q, adata, ndata)
    t.finished.connect(endtimer)
    t.start()

    sys.exit(app.exec_())


def structuredfscplot(fsc, pomdp, nepisodes):
    alabels = tuple(pomdp.aspace.values)
    nlabels = tuple(fsc.nspace.values)
    olabels = tuple(pomdp.ospace.values)
    amask = fsc.amask
    nmask = fsc.nmask

    q = mp.Queue()
    p = mp.Process(target=process_target, args=(q, nepisodes, alabels, nlabels, olabels, amask, nmask))
    p.daemon = True
    p.start()
    return q, p

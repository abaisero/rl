import sys
import multiprocessing as mp

from pyqtgraph.Qt import QtGui, QtCore
from .widgets import DistWidget, FSCWindow

import numpy as np


class DataThread(QtCore.QThread):
    def __init__(self, q, adata):
        super().__init__()
        self.q = q
        self.adata = adata

    def run(self):
        for idx, adist in iter(self.q.get, None):
            self.adata[idx] = adist[:, np.newaxis]


def process_target(q, nepisodes, alabels):
    nlabels = ['CF']

    na = len(alabels)
    nn = len(nlabels)

    ashape = nepisodes, na, nn
    adata = np.full(ashape, np.nan)

    app = QtGui.QApplication([])
    gui = FSCWindow().setup()
    gui.setWindowTitle('CF')

    gui.addTab(
        DistWidget().setup(adata, alabels, nlabels),
        'A-Strategy',
    )

    gui.update(updateAll=True)
    gui.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(gui.update)
    timer.start(1000)

    def endtimer():
        timer.stop()
        gui.update(updateAll=True)

    t = DataThread(q, adata)
    t.finished.connect(endtimer)
    t.start()

    sys.exit(app.exec_())


class CF_Plotter:
    def __init__(self, cf, nepisodes):
        self.cf = cf

        alabels = tuple(cf.aspace.values)

        self.q = mp.Queue()
        args = self.q, nepisodes, alabels
        self.p = mp.Process(target=process_target, args=args)
        self.p.daemon = True
        self.p.start()

        self.idx = 0

    def update(self, params):
        aprobs = self.cf.amodel.probs(params[0], ())

        self.q.put((self.idx, aprobs))
        self.idx += 1

    def close(self):
        self.q.put(None)

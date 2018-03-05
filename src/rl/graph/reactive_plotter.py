import multiprocessing as mp
import threading
import queue
import sys

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from .widgets import DistWidget, DistComboWidget, FSCWindow

import numpy as np


class DataThread(QtCore.QThread):
    def __init__(self, q, data):
        super().__init__()
        self.q = q
        self.data = data

    def run(self):
        for idx, a0dist, adist in iter(self.q.get, None):
            self.data[idx, :, 0] = a0dist
            self.data[idx, :, 1:] = adist.T


def process_target(q, nepisodes, alabels, olabels):
    na = len(alabels)
    no = len(olabels)

    olabels.insert(0, '*')

    shape = nepisodes, na, no + 1
    data = np.full(shape, np.nan)

    app = QtGui.QApplication([])
    gui = DistWidget().setup(data, alabels, olabels)
    gui.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(gui.update)
    timer.start(1000)

    def endtimer():
        timer.stop()
        gui.update()

    t = DataThread(q, data)
    t.finished.connect(endtimer)
    t.start()

    sys.exit(app.exec_())


def reactiveplot(reactive, nepisodes):
    alabels = reactive.env.afactory.values
    olabels = reactive.env.ofactory.values

    q = mp.Queue()
    p = mp.Process(target=process_target, args=(q, nepisodes, alabels, olabels))
    p.daemon = True
    p.start()
    return q, p

import multiprocessing as mp
import threading
import queue
import sys

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from .widgets import PPlotWidget

import numpy as np


class DataThread(QtCore.QThread):
    def __init__(self, q, data):
        super().__init__()
        self.q = q
        self.data = data

    def run(self):
        for idx, value in iter(self.q.get, None):
            self.data.itemset(idx, value)


def process_target(shape, q, pdict, **kwargs):
    data = np.full(shape, np.nan)

    app = QtGui.QApplication([])
    gui = PPlotWidget().setup(data, pdict, **kwargs)
    gui.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(gui.update)
    timer.start(1000 / 10)

    def endtimer():
        timer.stop()

    t = DataThread(q, data)
    t.finished.connect(endtimer)
    t.start()

    sys.exit(app.exec_())


def pplot(shape, pdict, **kwargs):
    q = mp.Queue()
    p = mp.Process(target=process_target, args=(shape, q, pdict), kwargs=kwargs)
    p.daemon = True
    p.start()
    return q, p

import sys
import multiprocessing as mp

from pyqtgraph.Qt import QtGui, QtCore
from .widgets import PlotWidget

import numpy as np


class DataThread(QtCore.QThread):
    def __init__(self, q, data):
        super().__init__()
        self.q = q
        self.data = data

    def run(self):
        for idx, value in iter(self.q.get, None):
            self.data[..., idx] = value
            # self.data.itemset(idx, value)


def process_target(q, shape, **kwargs):
    data = np.full(shape, np.nan)

    app = QtGui.QApplication([])
    gui = PlotWidget().setup(data, **kwargs)
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


class Plotter:
    def __init__(self, shape, **kwargs):
        self.q = mp.Queue()
        args = self.q, shape
        self.p = mp.Process(target=process_target, args=args, kwargs=kwargs)
        self.p.daemon = True
        self.p.start()

        self.idx = 0

    def update(self, data):
        self.q.put((self.idx, data))
        self.idx += 1

    def close(self):
        self.q.put(None)

import sys
import multiprocessing as mp

from pyqtgraph.Qt import QtGui, QtCore
from .widgets import DistWidget

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

    olabels = ('*',) + olabels

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


class Reactive_Plotter:
    def __init__(self, reactive, nepisodes):
        self.reactive = reactive

        alabels = tuple(reactive.aspace.values)
        olabels = tuple(reactive.ospace.values)

        self.q = mp.Queue()
        args = self.q, nepisodes, alabels, olabels
        self.p = mp.Process(target=process_target, args=args)
        self.p.daemon = True
        self.p.start()

        self.idx = 0

    def update(self, params):
        a0probs = self.reactive.a0model.probs(params[0], ())
        aprobs = self.reactive.amodel.probs(params[1], ())

        self.q.put((self.idx, a0probs, aprobs))
        self.idx += 1

    def close(self):
        self.q.put(None)

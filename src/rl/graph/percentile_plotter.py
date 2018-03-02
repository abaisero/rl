import multiprocessing as mp
import threading
import queue

import numpy as np


def thread_target(l, q, data):
    for idx, value in iter(q.get, None):
        # with l:
        data.itemset(idx, value)


def process_target(shape, q, pdict, **kwargs):
    data = np.full(shape, np.nan)
    l = mp.Lock()

    t = threading.Thread(target=thread_target, args=(l, q, data))
    t.deamon = True
    t.start()

    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
    from pyqtgraph.ptime import time

    app = QtGui.QApplication([])
    win = pg.GraphicsWindow()

    lab = win.addLabel(row=0, col=0, **kwargs.get('window', {}))

    percentiles = list(pdict)
    plt_pp = win.addPlot(
        title='Run Percentiles',
        left=kwargs.get('labels', {}).get('left', None),
        row=1, col=0)
    plt_pp.showAxis('right')
    plt_pp.addLegend(offset=(-30, -30))
    plt_pp.showGrid(x=True, y=True)


    curves_pp = {p: plt_pp.plot(**pdata) for p, pdata in pdict.items()}

    # TODO change to pg colormap
    import matplotlib.cm as cm
    cmap = cm.tab10_r
    plt_rp = win.addPlot(
        title='Individual Runs',
        left=kwargs.get('labels', {}).get('left', None),
        bottom=kwargs.get('labels', {}).get('bottom', None),
        row=2, col=0)

    plt_rp.showAxis('right')
    plt_rp.showGrid(x=True, y=True)
    curves_rp = [plt_rp.plot(pen=cmap(i%cmap.N, bytes=True)) for i in range(data.shape[0])]
    # TODO too many curves will break this

    # def setRange(rect=None, xRange=None, yRange=None, *args, **kwds):
    #     if not kwds.get('disableAutoRange', True):
    #         if yRange is not None:
    #             yRange[0] = 0
    #     # pg.ViewBox.setRange(vb, rect, xRange, yRange, *args, **kwds)
    # plt_rp.vb.setRange = setRange

    plt_rp.setYLink(plt_pp)
    plt_rp.setXLink(plt_pp)

    try:
        plt_rp.setRange(
            xRange=kwargs.get('ranges', {}).get('x'),
            yRange=kwargs.get('ranges', {}).get('y'),
        )
        # plt_rp.setLimits(
        #     xRange=kwargs.get('ranges', {}).get('x'),
        #     yRange=kwargs.get('ranges', {}).get('y'),
        # )
    except Exception:
        pass

    def update():
        # with l:
        nanpercentiles = np.nanpercentile(data, percentiles, axis=0)
        isnan = np.isnan(data)
        anynan = isnan.any(axis=1)
        allnan = isnan.all(axis=1)

        for p, pdata in zip(percentiles, nanpercentiles):
            curves_pp[p].setData(pdata)

        for c, d, anyn, alln in zip(curves_rp, data, anynan, allnan):
            if anyn and not alln:
                c.setVisible(True)
                c.setData(d)
            else:
                c.setVisible(False)

        # app.processEvents()  ## force complete redraw for every plot

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000 / 10)

    QtGui.QApplication.instance().exec_()


def pplot(shape, pdict, **kwargs):
    q = mp.Queue()
    p = mp.Process(target=process_target, args=(shape, q, pdict), kwargs=kwargs)
    p.daemon = True
    p.start()
    return q, p

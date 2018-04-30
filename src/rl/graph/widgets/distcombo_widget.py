from PyQt5 import QtWidgets
from .design.distcombo_widget import Ui_Form

import numpy as np


class DistComboWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.distWidget = self.ui.graphicsView
        self.comboBox = self.ui.comboBox

    def setup(self, data, ylabels, xlabels, wlabels, cidx, mask=None):
        if mask is None:
            mask = np.ones(data.shape[1:], dtype=np.bool)

        self.data = data
        self.mask = mask

        self.cidx = cidx
        self.widx = 0

        self.distWidget.setup(self.dw_data, ylabels, xlabels)

        self.comboBox.addItems(wlabels)
        self.comboBox.currentIndexChanged.connect(self.comboChanged)
        self.comboChanged(0)

        return self

    def comboChanged(self, i):
        self.widx = i
        self.distWidget.setData(self.dw_data)
        if self.mask is not None:
            self.distWidget.setMask(self.dw_mask)
        self.update()

    @property
    def dw_mask(self):
        idx = tuple(self.widx if i == self.cidx - 1 else slice(None)
                    for i in range(3))
        return self.mask[idx]

    @property
    def dw_data(self):
        idx = tuple(self.widx if i == self.cidx else slice(None)
                    for i in range(4))
        return self.data[idx]

    def update(self):
        self.distWidget.update()

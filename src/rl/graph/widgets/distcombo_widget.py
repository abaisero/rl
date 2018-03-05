from PyQt5 import QtWidgets
from .design.distcombo_widget import Ui_Form
from .dist_widget import DistWidget


class DistComboWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.distWidget = self.ui.graphicsView
        self.comboBox = self.ui.comboBox

    def setup(self, data, ylabels, xlabels, wlabels, cidx):
        self.data = data
        self.cidx = cidx
        self.widx = 0

        self.distWidget.setup(self.dw_data, ylabels, xlabels)

        self.comboBox.addItems(wlabels)
        self.comboBox.currentIndexChanged.connect(self.indexChanged)

        return self

    def indexChanged(self, i):
        self.widx = i
        self.distWidget.setData(self.dw_data)
        self.update()

    @property
    def dw_data(self):
        idx = tuple(self.widx if i == self.cidx else slice(None) for i in range(4))
        return self.data[idx]

    def update(self):
        self.distWidget.update()

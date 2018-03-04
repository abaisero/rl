from PyQt5 import QtWidgets
from .design.distplot_stackful import Ui_Form


class DistPlot_Stackful(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.stackedWidget = self.ui.stackedWidget
        self.comboBox = self.ui.comboBox
        self.comboBox.currentIndexChanged.connect(self.indexChanged)

    def addWidget(self, widget, name):
        self.stackedWidget.addWidget(widget)
        self.comboBox.addItem(name)

    def indexChanged(self, i):
        self.stackedWidget.setCurrentIndex(i)

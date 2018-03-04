from PyQt5 import QtWidgets
from .design.distplot_window import Ui_MainWindow


class FSC_Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.tabWidget = self.ui.tabWidget

    def addTab(self, widget, name):
        self.tabWidget.addTab(widget, name)

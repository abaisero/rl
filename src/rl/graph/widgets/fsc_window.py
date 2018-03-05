from PyQt5 import QtWidgets
from .design.distplot_window import Ui_MainWindow


class FSCWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.tabWidget = self.ui.tabWidget

    def addTab(self, widget, name):
        self.tabWidget.addTab(widget, name)

    def update(self, *, updateAll=False):
        if updateAll:
            for wi in range(len(self.tabWidget)):
                self.tabWidget.widget(wi).update()
        else:
            self.tabWidget.currentWidget().update()

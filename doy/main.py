from PyQt5 import QtWidgets, uic
import os
import sys

UiMainWindow, QtBaseClass = uic.loadUiType(os.path.join(os.path.dirname(__file__), "main.ui"))


class Doy(QtWidgets.QMainWindow, UiMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Doy()
    window.show()
    sys.exit(app.exec_())

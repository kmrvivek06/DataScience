import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton, QLabel
from PyQt5.QtGui import QFont, QPalette, QBrush,QPixmap
from PyQt5 import uic

class Dialog(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.resize(300, 300)
        self.setWindowTitle("New Window")
        self.label = QLabel(self)

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi("MainWindow.ui",self)
        self.resize(600, 600)
        self.setWindowTitle("Main Window-Python Code")
        qfont = QFont('Arial',12,QFont.Bold)
        self.setFont(qfont)
        self.new_button = QPushButton(self)
        self.new_button.setText("Open new window")
        self.new_button.resize(300, 40)
        self.new_button.setStyleSheet('background-color: #000; color: #fff;')
        self.dialog = Dialog()
        self.new_button.clicked.connect(self.openDialog)

    def openDialog(self):
        self.dialog.label.setText("New window opened from the Main Window")
        self.dialog.setWindowTitle("New Dialog")
        self.dialog.setStyleSheet('background-color: #ffff00;')
        self.dialog.exec_()    

app = QApplication(sys.argv)
_window = MainWindow()
palette = QPalette()
palette.setBrush(QPalette.Background,QBrush(QPixmap("../Pics/blue-squares-600.jpg")))
_window.setPalette(palette)
_window.show()
app.exec_()

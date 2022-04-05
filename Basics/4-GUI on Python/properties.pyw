import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QFont, QPalette, QBrush,QPixmap
from PyQt5 import uic

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

app = QApplication(sys.argv)
_window = MainWindow()
palette	= QPalette()
palette.setBrush(QPalette.Background,QBrush(QPixmap("../Pics/blue-squares-600.jpg")))
_window.setPalette(palette)
_window.show()
app.exec_()
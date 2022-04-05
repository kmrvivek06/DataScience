import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox
from PyQt5.QtGui import QIcon

class Window(QMainWindow):
  def __init__(self):
    QMainWindow.__init__(self)
    self.resize(800, 500)
    self.setStyleSheet('background-color: #55ff7f;')
    #Status Bar
    self.statusBar().showMessage("Welcome back!!")

    #menuBar Object
    menu = self.menuBar()
    #Parent Menu
    menu_File = menu.addMenu("&File")
    menu_Edit = menu.addMenu("&Edit")

    #Add action element to menu File
    menu_File_Open = QAction(QIcon("../Pics/pie16.png"), "&Open", self)
    menu_File_Open.setShortcut("Ctrl+o") #Shortcut
    menu_File_Open.setStatusTip("Open") #Message into the status bar
    menu_File_Open.triggered.connect(self.menuFileOpen) #Launcher
    menu_File.addAction(menu_File_Open)

    #Add action element to menu File
    menu_File_Close = QAction(QIcon(), "&Close", self)
    menu_File_Close.setShortcut("Ctrl+w") #Shorcut
    menu_File_Close.setStatusTip("Close") #Message into the status bar
    menu_File_Close.triggered.connect(self.menuFileClose) #Launcher
  
    menu_File.addAction(menu_File_Close)

    #Adding one sub menu to the Edit menu
    menu_Edit_Options = menu_Edit.addMenu("&Options")
  
    menu_Edit_Options_Search = QAction(QIcon(), "&Search", self)
    menu_Edit_Options_Search.setShortcut("Ctrl+f") #shorcut
    menu_Edit_Options_Search.setStatusTip("Search") #Message into the status bar
    menu_Edit_Options_Search.triggered.connect(self.menuEditOptionsSearch)

    menu_Edit_Options.addAction(menu_Edit_Options_Search)

  def menuFileOpen(self):
      QMessageBox.information(self, "Open", "Action Open", QMessageBox.Discard)
  
  def menuFileClose(self):
      QMessageBox.information(self, "Close", "Action Close", QMessageBox.Discard)
  
  def menuEditOptionsSearch(self):
      QMessageBox.information(self, "Search", "Action Search", QMessageBox.Discard)


app = QApplication(sys.argv)
window = Window()
window.show()
app.exec_()
  
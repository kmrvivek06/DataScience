import sys
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5 import uic


class Dialog(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    uic.loadUi("widgets.ui", self)
    self.setStyleSheet('background-color: #55ff7f;')
    #Radio buttons
    self.audi.toggled.connect(self.radio_value)
    self.bmw.toggled.connect(self.radio_value)
    self.ferrari.toggled.connect(self.radio_value)

    #Checkbox
    self.agree.toggled.connect(self.checkbox_state)
    self.disagree.toggled.connect(self.checkbox_state)

    self.combo.activated.connect(self.getItem)
    #For adding an item
    #self.combo.addItem("New car")
  
    #For delete an item
    #self.combo.removeItem(0)


  def radio_value(self):
    if self.audi.isChecked():
      self.label.setText("Audi is selected")
    elif self.bmw.isChecked():
      self.label.setText("BMW is selected")
    elif self.ferrari.isChecked():
      self.label.setText("Ferrari is selected")
    else:
      self.label.setText("No car is selected")
      

  def checkbox_state(self):
    if self.agree.isChecked():
      self.label.setText("You are agree..!!!")
    elif self.disagree.isChecked():
      self.label.setText("You are not agree..!!!")
    else:
      self.label.setText("")
      
  def getItem(self):
    item = self.combo.currentText()
    self.label.setText("Combo item is: " + item)    


app = QApplication(sys.argv)
dialog = Dialog()
dialog.show()
app.exec_()       
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QMessageBox, QTableWidget, QTableWidgetItem
from PyQt5 import uic
import pymysql
import pandas as pd

class Dialog(QDialog):
 def __init__(self):
  QDialog.__init__(self)
  self.setWindowTitle("Users Database") 
  self.resize(800, 600) #Initial size
  self.layout = QGridLayout() 
  self.setLayout(self.layout) #agregate the layout to the dialog
  self.table = QTableWidget() #create a table
  self.layout.addWidget(self.table) #agregate the table to the layout
  self.read_data()


 def read_data(self):
    try:
      conn = pymysql.connect(host='127.0.0.1',
                               port=3306,
                               user='python',
                               password='mysql',
                               db='customers')

      sql = "select name,sex,state,tel,addr from client join address on client.idclient = address.idcli"
      df=pd.read_sql(sql,con=conn)

      #make datagrid view/table
      self.table.setColumnCount(5)
      self.table.setHorizontalHeaderLabels(['Name', 'Sex', 'State', 'Tel', 'Address'])

      for i in range(len(df)):
            row = df.iloc[i]
            self.table.insertRow(i)
            name = QTableWidgetItem(str(row[0]))
            sex = QTableWidgetItem(str(row[1]))
            state = QTableWidgetItem(str(row[2]))
            tel = QTableWidgetItem(str(row[3]))
            address = QTableWidgetItem(str(row[4]))

            self.table.setItem(i, 0, name)
            self.table.setItem(i, 1, sex)
            self.table.setItem(i, 2, state)
            self.table.setItem(i, 3, tel)
            self.table.setItem(i, 4, address)

    finally:
        conn.close()   
        

app = QApplication(sys.argv)
dialog = Dialog()
dialog.show()
app.exec_()            

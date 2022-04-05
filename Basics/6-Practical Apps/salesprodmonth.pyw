from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow,QMessageBox,QApplication, QDialog,QTableWidgetItem
from PyQt5 import uic
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from connbd import connectdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Salesprodmonth(QDialog):
 def __init__(self,fn=None,parent=None):
#   QDialog.__init__(self)
    super(Salesprodmonth,self).__init__(parent,\
                                   flags=Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowCloseButtonHint)
    uic.loadUi("salesprodmonth.ui",self)
    self.btn_send.clicked.connect(self.salesxprod)

 def salesxprod(self):
    month = str(self.cmb_month.currentText())
    year = str(self.txt_year.text())
    
    #conn=connectdb()   
    #sql = "select codigo, facturas.umed, sum(cantidad) as Cantidad, sum(precioun*cantidad) as Total, (select max(costoun) from producto where facturas.codigo = producto.codprod and facturas.loteprod = producto.lote) as Costoun from facturas where mes ='"+ mes +"' and anio = '"+ anio +"' group by codigo"
    titles='Codigo Umed Cantidad Total CostoUn Util Rentab(%)'.split()  
    #df=pd.read_sql(sql,con=conn)
    #df.to_excel('../data/salesxprod.xls')
    df=pd.read_excel('../data/salesxprod.xls')
    
    df['Util']=(df['Total']-df['Cantidad']*df['Costoun'])
    df['Rentab(%)'] = ((df['Util'] / df['Total'])*100).apply(lambda x: round(x,2))
    
    df.columns = titles
    #conn.close()
    
    df['Cantidad']=df['Cantidad'].astype(int)

    #make datagrid view/table
    self.table.setColumnCount(7)
    self.table.setHorizontalHeaderLabels(['Codigo', 'Umed', 'Cantidad', 'Total', 'CostoUn', 'Util', 'Rentab(%)'])

    for i in range(len(df)):
            row = df.iloc[i]
            self.table.insertRow(i)
            codigo = QTableWidgetItem(str(row[0]))
            umed = QTableWidgetItem(str(row[1]))
            cantidad = QTableWidgetItem(str(row[2]))
            total = QTableWidgetItem(str(row[3]))
            costoun = QTableWidgetItem(str(row[4]))
            util = QTableWidgetItem(str(row[5]))
            rentab = QTableWidgetItem(str(row[6]))
            self.table.setItem(i, 0, codigo)
            self.table.setItem(i, 1, umed)
            self.table.setItem(i, 2, cantidad)
            self.table.setItem(i, 3, total)
            self.table.setItem(i, 4, costoun)
            self.table.setItem(i, 5, util)
            self.table.setItem(i, 6, rentab)

    #printing in matplotlib
    x=np.arange(len(df['Codigo']))
    df=df.sort_values(by='Total',ascending=False)
    y=df['Total']
    z=df['Util']
    
   
    b1=plt.bar(x,y,align='center',color='blue')
    b2=plt.bar(x,z,align='center',color='red')
    plt.xticks(x,df['Codigo'],rotation='vertical', fontsize='9')
    plt.title("Ventas y Utilidades-Periodo: %s - %s" % (month, year))
    
    plt.legend([b1,b2],["Ventas","Utilidades"],loc='upper right')          
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel('S/.')
    plt.show()
        
   
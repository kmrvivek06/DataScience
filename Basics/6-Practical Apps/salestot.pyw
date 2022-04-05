from connbd import connectdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np
import seaborn as sns
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QDialog,QTableWidgetItem,QMessageBox,QWidget
from PyQt5 import uic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,SGDRegressor
from sklearn import metrics,ensemble,svm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.kernel_ridge import KernelRidge

class SalesTot(QDialog):
  def __init__(self,fn=None,parent=None):
    # QDialog.__init__(self)
     super(SalesTot,self).__init__(parent,flags=Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint| Qt.WindowCloseButtonHint)
     uic.loadUi("salestot.ui",self)
     self.sales_util_tot()
     self.btnvtagiro.clicked.connect(self.grafvtagiro)
     self.btngrafvtatot.clicked.connect(self.grafvtatot)
     self.btncorr.clicked.connect(self.grafcorr)
     self.btnregresion.clicked.connect(self.regressionsales)
     self.btnpoly.clicked.connect(self.advanced_reg)    

  def data_read(self):
     self.dfsales2010_ = pd.read_excel('../data/salesdata2010-2014.xls',skiprows=[0], sheetname='Data',header=None)
     self.dfsales2010_.columns = 'Period Totals NTicket Sutures'.split()
     self.dfsales2010_['index']=self.dfsales2010_.index.values
    
     self.CR = pd.read_excel('../data/currency_rate.xls',skiprows=[0,1], sheetname='Monthly',header=None)
     self.CR.columns = 'Period CR'.split()
     self.CR = self.fixperiod(self.CR)
     self.CR = self.CR.loc['2011/01':'2016/12']
     self.CR.reset_index(inplace=True)
    
     self.dfmed = pd.read_excel('../data/Meds_2004-2016.xls',skiprows=[0,1], sheetname='Monthly',header=None)
     self.dfmed.columns = 'Period Value'.split()
     self.dfmed = self.fixperiod(self.dfmed)
     self.dfmed=self.dfmed.loc['2011/01':'2016/12']
     self.dfmed.reset_index(inplace=True)
    
     self.dfsales2010_.set_index(['Period'],drop=True,inplace=True)
     self.datat = self.dfsales2010_.loc['2011/01':'2016/12']
     self.datat.reset_index(inplace=True) 
     self.datat['AvgTicket']=self.datat['Totals']/self.datat['NTicket']
    
     self.datat['med']=self.dfmed['Value']
     self.datat['CR']=self.CR['CR']    
     self.dfsales2010_.reset_index(inplace=True)


  def fixperiod(self,dataf):
     monthnum = {'Ene':'01','Feb':'02','Mar':'03','Abr':'04','May':'05','Jun':'06','Jul':'07','Ago':'08','Sep':'09','Oct':'10','Nov':'11','Dic':'12'}
     month = dataf['Period'].apply(lambda x: monthnum.get(x[:-2]))
     year = dataf['Period'].apply(lambda x: '20'+x[3:])
     dataf['Period']=year + '/'+month
     dataf.set_index(['Period'],drop=True,inplace=True)
     return dataf    

  def normalize_data(self):
     #Transforming data into the same shape       
     self.x = self.dfsales['Period'].unique()  
     self.x = sorted(self.x)
     self.mercados = self.mercados.set_index(['Period'],drop=True)
     self.distprov = self.distprov.set_index(['Period'],drop=True)
     self.distlima = self.distlima.set_index(['Period'],drop=True)
     self.catering = self.catering.set_index(['Period'],drop=True)
         
     for i in np.arange(len(self.x)):
        if self.x[i] not in self.mercados.index:
            self.mercados.loc[self.x[i]]=[0,0,0]          
        elif self.x[i] not in self.distprov.index:
            sself.distprov.loc[self.x[i]]=[0,0,0] 
        elif self.x[i] not in self.catering.index:
            self.catering.loc[self.x[i]]=[0,0,0] 
        elif self.x[i] not in self.distlima.index:
            self.distlima.loc[self.x[i]]=[0,0,0] 
            
     self.mercados.reset_index(drop=False,inplace=True)  
     self.mercados.sort_values(by='Period', inplace=True,ascending=True)
     self.mercados.reset_index(drop=True,inplace=True)    
     self.catering.reset_index(drop=False,inplace=True)       
     self.catering.sort_values(by='Period', inplace=True, ascending=True)
     self.catering.reset_index(drop=True,inplace=True)
     self.distprov.reset_index(drop=False,inplace=True)         
     self.distprov.sort_values(by='Period', inplace=True, ascending=True)
     self.distprov.reset_index(drop=True,inplace=True)
     self.distlima.reset_index(drop=False,inplace=True)
     self.distlima.sort_values(by='Period', inplace=True, ascending=True)
     self.distlima.reset_index(drop=True,inplace=True)         


  def grafvtatot(self):
     #SEABORN MODULE
     sns.set_palette("GnBu_d")
     sns.set_style('whitegrid')
     
     g=sns.jointplot(x='index',y='Total',data=self.salestot,kind="reg",size=6)
     g.ax_joint.set_xticklabels(self.salestot['Period'],rotation=90,fontsize=7,stretch='condensed')    
     plt.xlabel('Period', fontsize=15)
     plt.ylabel('Total Sales S/.', fontsize=15) 
     plt.show()    
     g = sns.FacetGrid(self.dfsales,col='Giro',size=2, aspect=1.5)
     xplot = self.dfsales.index.values
     g.map(sns.regplot,"index","Total")     
     plt.show()
    
     
  
  def grafvtagiro(self):
     #GRAPHS BY SEGMENT
     y = [self.dfsales['Total'],self.distlima['Total'],self.distprov['Total'],self.catering['Total'],self.mercados['Total']]
     fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,8))
        
     axs[0].set_yticks([0,25000,50000,100000])
     axs[1].set_yticks([0,25000,50000,100000])
     axs[2].set_yticks([0,15000,30000,50000])
     axs[3].set_yticks([0,5000,10000,15000,30000])
     
     axs[0].plot(np.arange(len(self.distlima)),y[1],lw=5, color='blue')
     axs[1].plot(np.arange(len(self.distprov)),y[2],lw=5, color='blue')
     axs[2].plot(np.arange(len(self.catering)),y[3],lw=5, color='blue')
     axs[3].plot(np.arange(len(self.mercados)),y[4],lw=5, color='blue')
     ynayear = ["DistLima","DistProv","Catering","Mercados"]
      
    
     for n,ax in enumerate(axs):
            ax.set_xticks(np.arange(len(self.x)))
            ax.set_xticklabels(self.x,rotation=90,fontsize=8,stretch='condensed')
            ax.set_title(ynayear[n])
      
    
     plt.show()
        
     #ALL DATA IN ONE GRAPH   
     fig1, ax1 = plt1.subplots(1,1,figsize=(10,4))
     ax1.plot(np.arange(len(self.distlima)),y[1],label="DistLima",lw=5, color='blue')
     ax1.plot(np.arange(len(self.distprov)),y[2],label="DistProv",lw=5,color='red')
     ax1.plot(np.arange(len(self.catering)),y[3],label="Catering",lw=5, ls=':',color='green')
     ax1.plot(np.arange(len(self.mercados)),y[4],label="Mercados",lw=5, ls='--',color='black')
     
     ax1.set_yticks([0,1000,2500,5000,10000,20000,30000,50000,60000,90000])
     ax1.set_xticks(np.arange(len(self.x)))
     ax1.set_xticklabels(self.x,rotation=90,fontsize=8,stretch='condensed')
     ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)    
     plt1.legend(loc="upper left")
     plt1.show()   


  def sales_util_tot(self):
    
     #conn=connectdb()
     #sql = "select concat(substring(fecha,7,4),'/',substring(fecha,4,2)) as Period,giro, sum(importe) as Total, sum(utilidad) as Util from listafact group by Period desc, giro order by str_to_date(Period, '%Y/%m') desc" 
     titles='Period Giro Total Util Rentab(%)'.split()
     #self.dfsales=pd.read_sql(sql,con=conn)
     #self.dfsales.to_excel('../data/salesxsegmentxperiod.xls')
     self.dfsales=pd.read_excel('../data/salesxsegmentxperiod.xls')
     self.dfsales['Rentab(%)'] = ((self.dfsales['Util'] / self.dfsales['Total'])*100).apply(lambda x: round(x,2))
     self.dfsales.columns = titles
   
     #conn.close()
     #making datagrid view/table
     self.tablesalestot.setColumnCount(5)
     self.tablesalestot.setHorizontalHeaderLabels(['Period', 'Giro', 'Total', 'Util', 'Rentab(%)'])
              
     for i in range(len(self.dfsales)):
            row = self.dfsales.iloc[i]
            self.tablesalestot.insertRow(i)
            Period = QTableWidgetItem(str(row[0]))
            giro = QTableWidgetItem(str(row[1]))
            total = QTableWidgetItem(str(row[2]))
            util = QTableWidgetItem(str(row[3]))
            rentab = QTableWidgetItem(str(row[4]))
            self.tablesalestot.setItem(i, 0, Period)
            self.tablesalestot.setItem(i, 1, giro)
            self.tablesalestot.setItem(i, 2, total)
            self.tablesalestot.setItem(i, 3, util)
            self.tablesalestot.setItem(i, 4, rentab)
            
     self.tablesalestot.resizeColumnsToContents()
    
     #SALES BY SEGMENT BY PERIOD
     self.distlima = self.dfsales[(self.dfsales['Giro']=='Dist Lima')][['Period','Total','Util','Rentab(%)']]
     self.distprov = self.dfsales[(self.dfsales['Giro']=='Dist Prov')][['Period','Total','Util','Rentab(%)']]
     self.catering = self.dfsales[(self.dfsales['Giro']=='Catering')][['Period','Total','Util','Rentab(%)']]
     self.mercados = self.dfsales[(self.dfsales['Giro']=='Mercados')][['Period','Total','Util','Rentab(%)']]
     self.normalize_data()
            
  
    #DATAFRAME FOR TOTAL SALES BY PERIOD    
     self.salestot = self.dfsales.groupby('Period').sum().unstack()
     self.salestot = self.salestot['Total'].reset_index()    
     self.salestot.columns = 'Period Total'.split()    
    
     self.sales_seg =pd.concat([self.salestot['Total'],self.distlima['Total'],self.distprov['Total'],self.catering['Total'],self.mercados['Total']],axis=1)
     self.sales_seg.columns = 'salesTot DistLima DistProv Catering Mercados'.split()
     self.sales_seg['Period'] = self.x
        
     self.tablesalesseg.setColumnCount(6)
     self.tablesalesseg.setHorizontalHeaderLabels(['Period', 'salesTot', 'DistLima', 'DistProv', 'Catering', 'Mercados'])
              
     for i in range(len(self.sales_seg)):
            row = self.sales_seg.iloc[i]
            self.tablesalesseg.insertRow(i)
            period = QTableWidgetItem(str(row[5])) 
            vtatot = QTableWidgetItem(str(row[0]))
            distlima = QTableWidgetItem(str(row[1]))
            distprov = QTableWidgetItem(str(row[2]))
            catering = QTableWidgetItem(str(row[3]))
            mercados = QTableWidgetItem(str(row[4]))
            
            self.tablesalesseg.setItem(i, 0, period)
            self.tablesalesseg.setItem(i, 1, vtatot)
            self.tablesalesseg.setItem(i, 2, distlima)
            self.tablesalesseg.setItem(i, 3, distprov)
            self.tablesalesseg.setItem(i, 4, catering)
            self.tablesalesseg.setItem(i, 5, mercados)
            
     self.tablesalesseg.resizeColumnsToContents()    


  def minmaxScaler(self,X_train,X_test,y_train,y_test):
     scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
     scaler.fit(X_train)
     X_train_scaled = scaler.transform(X_train)
     X_test_scaled = scaler.transform(X_test)
     scaler.fit(y_train)
     y_train_scaled = scaler.transform(y_train)
     y_test_scaled = scaler.transform(y_test)

     return X_train_scaled, X_test_scaled, y_train_scaled,y_test_scaled

  def stdScaler(self,X_train,X_test,y_train,y_test):
      X_scaler = StandardScaler()
      y_scaler = StandardScaler()
      X_train = X_scaler.fit_transform(X_train)
      y_train = y_scaler.fit_transform(y_train)
      X_test = X_scaler.transform(X_test)
      y_test = y_scaler.transform(y_test)

      return X_train, X_test, y_train, y_test

  def training(self,X,y):    
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
     X_train_scaled, X_test_scaled, y_train_scaled,y_test_scaled = self.stdScaler(X_train,X_test,y_train,y_test)
       
     return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

  def grafcorr(self):
     #REGRESSION WITH EXTERNAL VARIABLE: NATIONAL MEDS PRODUCTION
     df = pd.read_excel('../data/Meds_2004-2016.xls',skiprows=[0,1], sheetname='Monthly',header=None)
     df.columns = 'Period Value'.split()
     yearnum = {'Ene':'01','Feb':'02','Mar':'03','Abr':'04','May':'05','Jun':'06','Jul':'07','Ago':'08','Sep':'09','Oct':'10','Nov':'11','Dic':'12'}
     month = df['Period'].apply(lambda x: yearnum.get(x[:-2]))
     year = df['Period'].apply(lambda x: '20'+x[3:])
     df['Period']=year + '/'+month
     df.set_index(['Period'],drop=True,inplace=True)
     df=df.loc['2014/10':'2016/12']
     df.reset_index(inplace=True)
     self.sales_seg = self.sales_seg[:-5]    
     self.sales_seg['ProdMeds'] = df['Value']    
    
     sns.set(style="ticks", color_codes=True)
     sns.pairplot(self.sales_seg,kind="reg",size=1,aspect=2)
     plt.show()    

  def regressionsales(self):

     self.data_read()
        
     #SKLEARN LINEAR REGRESSION MODEL
     X = self.datat[['NTicket','AvgTicket','med','CR']]
     y = self.datat['Totals'] 
        
     X_train, X_test, y_train, y_test = self.training(X, y)    
     lm = LinearRegression()
     lm.fit(X_train,y_train)

     predictions = lm.predict(X_test)
     
     coeffecients = pd.DataFrame(lm.coef_,X.columns)
     coeffecients.columns = ['Coeffecient']
     
     score = metrics.r2_score(y_test, predictions)
     self.lbltext.setText('R-squared: %.4f' % score + '\n' +str(coeffecients))    
     self.lblcorr.setText('Correlation Matrix'+'\n'+str(self.datat[['Totals','NTicket','AvgTicket',  'med','CR']].corr()))
     
     #Graphing the relatioship between variables
     sns.set_palette("GnBu_d")
     sns.set_style('whitegrid')
     sns.set(style="ticks", color_codes=True)
     sns.pairplot(self.datat[['Totals','NTicket','AvgTicket','med','CR']],kind="reg",size=1.5,aspect=2)
     plt.show()           

  def advanced_reg(self):
     self.data_read()    
     y = self.datat['Totals']
     X = self.datat['NTicket']
     X = X.reshape(-1,1)
     y = y.reshape(-1,1)   
        
     X_train, X_test, y_train, y_test = self.training(X, y)
     xx=np.linspace(-3,3,100)

     #SGD Regressor:
     sgd = SGDRegressor(alpha=0.00001,penalty='elasticnet',l1_ratio=0.5,n_iter=1000)
     sgd.fit(X_train, y_train)
     ysgd=sgd.predict(xx.reshape(xx.shape[0],1))

     plt.figure(figsize=(10,8)) 
     plt.subplot(231)
     plt.title('SGDRegressor')
     plt.axis([-3,3,-3,3])
     plt.grid(True)
     plt.scatter(X_train, y_train) 
     #plt.xlabel('Ticket Numbers')
     plt.ylabel('Total Sales')
     plt.plot(xx,ysgd,c='g',linestyle='--')    

     #Linear Regression
     regl = LinearRegression(normalize=True)
     regl.fit(X_train,y_train)    
     yy = regl.predict(xx.reshape(xx.shape[0],1)) 

     plt.subplot(232)
     plt.title('Linear Regression')
     plt.axis([-3,3,-3,3])
     plt.grid(True)
     plt.scatter(X_train, y_train) 
     #plt.xlabel('Ticket Numbers')
     plt.ylabel('Total Sales')
     plt.plot(xx,yy)    

     #Polynomial Regression
     poly = PolynomialFeatures(degree=2)
     X_train_poly = poly.fit_transform(X_train)
     X_test_poly = poly.fit_transform(X_test)

     regpoly = LinearRegression(normalize=True)
     regpoly.fit(X_train_poly,y_train)
     xxpoly = poly.transform(xx.reshape(xx.shape[0],1))
     ypoly = regpoly.predict(xxpoly)

     plt.subplot(233)
     plt.title('Polynomial Regression')
     plt.axis([-3,3,-3,3])
     plt.grid(True)
     plt.scatter(X_train, y_train)
     #plt.xlabel('Ticket Numbers')
     plt.ylabel('Total Sales') 
     plt.plot(xx,ypoly,c='r',linestyle='--')

     #Lasso,Rigde and elastic net (L1 and L2 penalisation)
     en = ElasticNet(alpha=0.00001,l1_ratio=0.5,copy_X=True,normalize=True,max_iter=1000)
     en.fit(X_train,y_train)
     yen = en.predict(xx.reshape(xx.shape[0],1))

     plt.subplot(234)
     plt.title('ElasticNet l1+l2')
     plt.axis([-3,3,-3,3])
     plt.grid(True)
     plt.scatter(X_train, y_train) 
     plt.xlabel('Ticket Numbers')
     plt.ylabel('Total Sales')
     plt.plot(xx,yen,c='g',linestyle='--')

     #Random Forest regressor
     rfr = ensemble.ExtraTreesRegressor(n_estimators=10)
     rfr.fit(X_train,y_train)
     yrfr = rfr.predict(xx.reshape(xx.shape[0],1))
     plt.subplot(235)
     plt.title('Random Forest Regressor')
     plt.axis([-3,3,-3,3])
     plt.grid(True)
     plt.scatter(X_train, y_train) 
     plt.xlabel('Ticket Numbers')
     plt.ylabel('Total Sales')
     plt.plot(xx,yrfr,c='y',linestyle='--')


     #Support Vector Machines for linear regression
     rbf = svm.SVR(kernel='rbf', C=0.1, degree=3, gamma='auto')
     rbf.fit(X_train,y_train)
     yrbf = rbf.predict(xx.reshape(xx.shape[0],1))

     plt.subplot(236)
     plt.title('SVM Rbf')
     plt.axis([-3,3,-3,3])
     plt.grid(True)
     plt.scatter(X_train, y_train) 
     plt.xlabel('Ticket Numbers')
     plt.ylabel('Total Sales')
     plt.plot(xx,yrbf,c='b',linestyle='--')  
     plt.show()
     
     sgdscore = sgd.score(X_test,y_test)
     linealscore = regl.score(X_test,y_test)
     polyscore = regpoly.score(X_test_poly,y_test)
     regscore = en.score(X_test,y_test)
     rfrscore = rfr.score(X_test,y_test)
     rbfscore = rbf.score(X_test,y_test)
     self.lbltext.setText(
        'SGDRegressor R-squared: %.4f' % sgdscore +'\n'+
        'Lineal R-squared: %.4f' % linealscore + '\n' +
        'Polynomial R-squared: %.4f' % polyscore + '\n' +
        'ElasticNet score: %.4f' % regscore +'\n'+
        'Random Forest Regressor Score:%.4f' % rfrscore+'\n'+
        'SVM Rbf Score:%.4f' % rbfscore)          
    
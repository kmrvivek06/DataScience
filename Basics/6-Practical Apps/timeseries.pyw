import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QDialog,QTableWidgetItem,QMessageBox,QWidget
from PyQt5 import uic
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import pickle as pk
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf, adfuller
from pandas.tools.plotting import autocorrelation_plot

#Binaries can be installed in Anaconda
#conda install statsmodels

class TimeSeries(QDialog):
	def __init__(self,fn=None,parent=None):
		super(TimeSeries,self).__init__(parent,flags=Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowCloseButtonHint)
		uic.loadUi("timeseries.ui",self)
		self.loadData()
		self.btnarima.clicked.connect(self.Arima)
		self.btnarimap.clicked.connect(self.paramArima)
		self.btnother.clicked.connect(self.otherModels)

	def loadData(self):
		
		dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
		
		datatable = pd.read_excel('../data/outputs2011-.xls',sheetname='SN30TC2075')
		self.data = datatable.copy(deep=True)
		self.data['Month']=self.data['Month'].apply(dateparse)
		self.data.set_index('Month', inplace=True)
		self.data['Quantity'] = self.data['Quantity'].astype(np.float64)

		#making datagrid view/table
		self.tstable.setColumnCount(2)
		self.tstable.setHorizontalHeaderLabels(['Period','Quantity'])
		for i in range(len(datatable)):
			row = datatable.iloc[i]
			self.tstable.insertRow(i)
			period = QTableWidgetItem(str(row[0]))
			quantity = QTableWidgetItem(str(row[1]))
			self.tstable.setItem(i, 0, period)
			self.tstable.setItem(i, 1, quantity)

		#self.tstable.resizeColumnsToContents()	

	def paramArima(self):
		res = self.data - self.data.shift()
		res.dropna(inplace=True)
		#autocorrelation_plot(self.data['Quantity'])

		_acf = acf(res['Quantity'], nlags=20)
		_pacf = pacf(res['Quantity'], nlags=20, method='ols')

		#Plot ACF: 
		plt.subplot(121) 
		plt.plot(_acf)
		plt.axhline(y=0,linestyle='--',color='gray')
		plt.axhline(y=-1.96/np.sqrt(len(res)),linestyle='--',color='gray')
		plt.axhline(y=1.96/np.sqrt(len(res)),linestyle='--',color='gray')
		plt.title('Autocorrelation Function')

		#Plot PACF:
		plt.subplot(122)
		plt.plot(_pacf)
		plt.axhline(y=0,linestyle='--',color='gray')
		plt.axhline(y=-1.96/np.sqrt(len(res)),linestyle='--',color='gray')
		plt.axhline(y=1.96/np.sqrt(len(res)),linestyle='--',color='gray')
		plt.title('Partial Autocorrelation Function')
		plt.tight_layout()
		plt.show()	

	def Arima(self):
		res = self.data - self.data.shift(1)
		res.dropna(inplace=True)
	
		model = ARIMA(self.data, order=(1,1,1))
		model_fit = model.fit(disp=-1) # If True, convergence information is printed. disp < 0 means no output in this case.
		predictions = model_fit.predict(1,75,typ='levels')
		#SARIMAX
		modSAR = SARIMAX(self.data['Quantity'], order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 3),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

		resultsSAR = modSAR.fit()
		predictSAR = resultsSAR.predict(1,75,dynamic=False)
	
		self.data['forecast'] = predictions
		self.data['forecastSAR'] = predictSAR
		self.data[['Quantity','forecast','forecastSAR']].plot(figsize=(8,6))

		fig, ax = plt.subplots(3,1,figsize=(8,8))
		ax[0].plot(predictions,label="ARIMA Pred",lw=3, color='blue')
		ax[0].plot(predictSAR, label='SARIMAX Pred',lw=3,color='red')
		ax[0].scatter(self.data.index.values,self.data['Quantity'], color='darkorange', label='data')
		ax[0].set_ylabel('Quantity')
		ax[0].set_title('Arima Model Inventory')
		ax[0].legend(loc='best')

		ax[1].plot(res,color='green', label='Residuals')
		ax[1].plot(model_fit.fittedvalues, color='red', lw=3,label='Model fit')	
		ax[1].legend(loc='best')
		
		ax[2].plot(self.data['Quantity'].rolling(window=12,center=False).mean(),label="Mean Average",lw=3, color='green')
		ax[2].plot(self.data['Quantity'].rolling(window=12,center=False).std(),label="Std Average",lw=3, color='blue')
		ax[2].set_xlabel('Period')
		ax[2].legend(loc='best')
	
		self.lblmodel.setText(str(model_fit.summary()))
		plt.show()
		self.lblmodel1.setText(str(res.describe()))
		self.lblsvr.setText(str(model_fit.forecast(steps=3,alpha=0.05)[0]))
		self.test_stationarity(res['Quantity'])	


	def test_stationarity(self,timeseries):
		test = adfuller(timeseries, autolag='AIC')
		output = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
		for key,value in test[4].items():
			output['Critical Value (%s)'%key] = value

		self.lbldftest.setText(str(output))
		
	def otherModels(self):
		ydata = self.data['Quantity'].ravel()
		Xdata = np.arange(len(self.data.index.values)).reshape(-1,1)
		
		test_set = np.array([[76],[77],[78]])

		svr_rbf = SVR(kernel='rbf', C=1e3,gamma=0.1)
		svr_lin = SVR(kernel='linear', C=1e3)
		krr = KernelRidge(kernel='rbf', gamma=1)
		mlpr = MLPRegressor(hidden_layer_sizes=100,
							activation='relu',
							alpha=0.0001,
							batch_size='auto',
							solver='adam',
							learning_rate='constant',
							learning_rate_init=0.001,
							max_iter=300,
							shuffle=True)

		# fit your model with the training set
		y_rbf = svr_rbf.fit(Xdata, ydata).predict(Xdata)
		y_lin = svr_lin.fit(Xdata, ydata).predict(Xdata)
		y_krr = krr.fit(Xdata, ydata).predict(Xdata)
		y_mplr = mlpr.fit(Xdata, ydata).predict(Xdata)
		
		# #predict on a test set
		joblib.dump(mlpr, 'mlpr_ts.pkl')
		clf = joblib.load('mlpr_ts.pkl')
		self.lblsvr.setText(str(clf.predict(test_set)))

		plt.scatter(Xdata, ydata, color='darkorange', label='data')		
		plt.plot(Xdata, y_rbf, color='navy', lw=2, label='SVR-RBF model')
		plt.plot(Xdata, y_lin, color='green', lw=2, label='SVR-Linear model')
		plt.plot(Xdata, y_krr, color='red', lw=2, label='Kernel Ridge Regression')
		plt.plot(Xdata, y_mplr, color='yellow', lw=2,label='MPL Regressor' )
		plt.xlabel('Period')
		plt.ylabel('Quantity')
		plt.title('Regression Models Inventory')
		plt.legend(loc='best')
		plt.show()		

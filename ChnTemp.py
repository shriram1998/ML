import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

def PreprocessData(df):
    df['Date']=df['Date'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d"))
    df.set_index('Date',inplace=True)
    df=df[df['Temp']>0]
    ts=df['Temp']
    return ts

def test_stationarity(ts):
    rolmean = ts.rolling(365).mean()
    rolstd = ts.rolling(365).std()
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.draw()
    #Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def ModelParams(ts):
    lag_acf = acf(ts, nlags=20)
    lag_pacf = pacf(ts, nlags=20, method='ols')
    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.draw()

def BuildModel(ts):
    model = ARIMA(ts, order=(9, 0, 0))
    results_ARIMA = model.fit(disp=-1)
    return results_ARIMA

def EvaluateModel(model,ts):
    predicted=model.fittedvalues.shift(-1)[:-1]
    actual=ts[:-1]
    #shifting one day prediction because of common time series correlation issue
    plt.plot(actual[:20])
    plt.plot(predicted[:20], color='red')
    plt.title('RSS: %.4f'% sum((predicted-actual)**2))
    plt.draw()
    print(model.predict('2020-01-01','2020-01-25').apply(lambda x:(x-32)*5/9).shift(-1))

def main():
    df=pd.DataFrame(pd.read_csv("D:\ML course\ML Projects\ChnTemperaure\Temp.csv"))
    ts=PreprocessData(df)
    # test_stationarity(ts)
    #The series is stationary as Test statistic is less than critical value(1%)
    # ModelParams(ts)
    #Result:(9,0,0)
    model=BuildModel(ts)
    EvaluateModel(model,ts)
    plt.show()
    
if __name__ == "__main__":
    main()

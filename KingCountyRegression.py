import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn import metrics
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def CleanData(df):
    #Data cleaning
    df.drop(['id','zipcode'],axis=1,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    df['month'] = df['date'].apply(lambda date:date.month)
    df['year'] = df['date'].apply(lambda date:date.year)
    df.drop('date',axis=1,inplace=True)
    df = df.sort_values('price',ascending=False).iloc[216:]
    return df
def PreprocessData(df):
    #preprocessing
    X=df.drop('price',axis=1)
    y=df['price']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
    scaler=MinMaxScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train,X_test,y_train,y_test
def BuildModel(X_train,y_train):
    # Building the model
    model=Sequential()
    model.add(Dense(19,activation='relu'))
    model.add(Dense(19,activation='relu'))
    model.add(Dense(19,activation='relu'))
    model.add(Dense(19,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')

    model.fit(x=X_train,y=y_train,
              validation_data=(X_test,y_test),
              batch_size=128,epochs=500)
    pd.Series(model.history.history['loss']).plot()
    plt.draw()
    model.save('99%500epoch.h5')
    return model

def EvaluateModel(X_train, X_test, y_train, y_test,model):
    #Evaluating the model with new data
    pred=model.predict(X_test)
    plt.scatter(x=y_test,y=pred)
    plt.draw()
    print('MSE:',metrics.mean_squared_error(y_test,pred))
    print('RMSE:',metrics.mean_squared_error(y_test,pred)**0.5)
    print('Explained var score:',metrics.explained_variance_score(y_test,pred))
    print('R2 score:',metrics.r2_score(y_test,pred))
    print('Train score:', model.evaluate(X_train,y_train))
    print('Test score:', model.evaluate(X_test,y_test))
def main():
    df=pd.read_csv('./kc_house_data.csv')
    #Loading the model
    model=load_model('99%500epoch.h5')
    X_train,X_test,y_train,y_test=PreprocessData(CleanData(df))
    EvaluateModel(X_train,X_test,y_train,y_test,model)
    plt.show()

if __name__ == "__main__":
    main()
